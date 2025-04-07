import numpy as np
import pandas as pd
import os, sys, yaml
from datetime import datetime
import calendar
import torch
import xarray as xr
from pathlib import Path
import copy

#Custom imports
from .data_utils import (setup_matplotlib, interpolate_clima, ndvi as _ndvi, _process_mask, _standarize, 
                        _auto_scale, _scale, _npdate2str, plot_clima, plot_txy, plot_xy, plot_t, NDVI)
setup_matplotlib(); #Create custom colormap, set default styles
from .aggregateevents import aggregate_events

class DeepExtremes(torch.utils.data.Dataset):
    def __init__(self, config, subset='train', test_mode='testlist', debug=None):
        '''
            Test mode one of ['all', 'ranmdom', 'testlist']
        '''
        #Paths and dates
        self.datacubes_path= Path(config['data']['datacubes_path'])
        registry_path= Path(config['data']['registry_path'])
        registry_read_until= datetime.strptime(config['data']['registry_read_until'], '%m/%d/%Y')
        if config['data']['test_start_date'] is None:
            self.test_start_date= None
        else:
            self.test_start_date= datetime.strptime(config['data']['test_start_date'], '%m/%d/%Y')
        assert self.datacubes_path.exists(), f'{self.datacubes_path} not found'
        assert registry_path.exists(), f'{registry_path} not found'
        
        #If debug, save images and print data
        self.debug= config['debug'] if debug is None else debug #__init__ debug has preference
        if self.debug: (self.datacubes_path / 'debug').mkdir(exist_ok=True)
        
        #Process csv into DF
        self.registry= pd.read_csv(registry_path)
        self.registry= self.registry.set_index('location_id')
        def event_processer(e):
            try:    return eval(e)
            except: return [(-1, '', '')]
        #Processa events and geometry from str into python structures
        self.registry['events']= self.registry['events'].apply(event_processer)
        self.registry['geometry']= self.registry['geometry'].apply(lambda p: p.replace('POLYGON ', '')\
                                           .replace('(', '').replace(')', '').replace(',', '').split(' '))
        #Get lon and lat
        for label, idx in zip(['x1', 'y1', 'x2', 'y2'], [0,1,4,5]):
            self.registry[f'geometry_{label}']= self.registry['geometry'].apply(lambda a: float(a[idx]))
        #Keep specific cube versions
        versions= config['data']['versions']
        if versions is not None: self.registry= self.registry[self.registry.version.isin(versions)]
        #For replicability, keep only cubes that were generated up to a certain date
        self.registry['creation_date']= pd.to_datetime(self.registry['creation_date'])#, format='%d%b%Y:%H:%M:%S.%f')
        self.registry['modification_date']= pd.to_datetime(self.registry['modification_date'])
        self.registry= self.registry[self.registry.modification_date < registry_read_until]        
        
        #Get a list of all downloaded minicube ids and keep only those that are also in the csv (just to be sure)
        cubes_list= list(self.datacubes_path.glob('deepextremes-minicubes/*/*/*.zarr'))
        assert len(cubes_list), f'No files found at {self.datacubes_path} with pattern "*/*.zarr"'
        cubes_list= ['/'.join(c.parts[-4:]) for c in cubes_list]
        self.registry= self.registry[self.registry.path.isin(cubes_list)]
        
        #Load all info from config file    
        #Get only training / val data
        train_split= config['data']['split']['train']
        val_split= config['data']['split']['val']
        test_split= config['data']['split']['test']
        assert train_split + val_split + test_split == 1
        self.subset= subset
        self.registry= self.registry.sample(frac=1, random_state=config['seed']) #Shuffle registry
        N= int(len(self.registry))
        
        #If there is a testlist, move those cubes at the end, so that they are always taken by the test set
        testlist_file= config['data'].get('testlist_path', None)
        testlist_csv= config['data'].get('testlist_csv', None)
        if testlist_file is not None and Path(testlist_file).exists() and self.subset != 'all': 
            self.testlist= [cube.split('/')[-1].replace('.zarr', '') for cube in open(Path(testlist_file)).read().splitlines()]
            if self.debug: print(f'Using {testlist_file=} with {len(self.testlist)} items')
        elif testlist_csv is not None and Path(testlist_csv).exists() and self.subset != 'all': 
            test_df= pd.read_csv(testlist_csv, header=0)
            fold= config['data'].get('testlist_fold', 5)
            self.testlist= list(test_df[test_df.group == fold].mc_id.values)
            if self.debug: print(f' > Using fold {fold} from test list {testlist_csv} with {len(self.testlist)} items')
        else:
            self.testlist= []
        if len(self.testlist):
            #We put the testlist items at the end of the registry
            self.registry= pd.concat([self.registry[~self.registry.mc_id.isin(self.testlist)],
                                      self.registry[self.registry.mc_id.isin(self.testlist)]])
            assert int(N*test_split) > len(self.testlist), f'Test fraction {test_split=} is too small'+\
                f' ({int(N*test_split)}) to hold all cubes ({len(self.testlist)}) in {testlist_file=} / {testlist_csv=}'
                
        #Separate in subsets
        if self.subset == 'train': 
            self.registry= self.registry.iloc[:int(N*train_split)]
        elif self.subset == 'val': 
            self.registry= self.registry.iloc[int(N*train_split):int(N*(train_split+val_split))]
        elif self.subset == 'test': 
            if test_mode == 'all':
                self.registry= self.registry.iloc[int(N*(train_split+val_split)):int(N*(train_split+val_split+test_split))]
            elif test_mode == 'testlist':
                self.registry= self.registry.iloc[-len(self.testlist):]
            elif test_mode == 'random':
                self.registry= self.registry.iloc[int(N*(train_split+val_split)):-len(self.testlist)]
            else:
                test_modes= ['all', 'ranmdom', 'testlist']
                raise AssertionError(f'{test_mode=} must be in {test_modes}')
        elif self.subset == 'all': 
            self.registry= self.registry
        else: 
            raise AssertionError(f"{subset=} must be in ['train', 'val', 'test', 'all']")
            
        #We do this after the subset so that we can update the blacklist without affecting the data in each set
        #Get a blacklist of minicubes and delete from the registry all cubes in this file
        blacklist_file= Path(config['data']['blacklist_path'])
        if blacklist_file.exists(): 
            blacklist= [(cube.split('/')[-1] if '/' in cube else cube).replace('.zarr', '') 
                            for cube in open(blacklist_file).read().splitlines() if len(cube) > 10]
            if debug: print(f'Using {blacklist_file=} with {len(blacklist)} items')
            self.registry= self.registry[~self.registry.mc_id.isin(blacklist)]
        
        #Load feature names to read
        self.txy_real= config['data']['variables']['txy_real']
        self.txy_cat= config['data']['variables']['txy_cat'] #['SCL'] #Do not use SCL at all for now
        self.txy_mask= config['data']['variables']['txy_mask'] #Special _cat that is used for global valid masking
        self.xy_cat= config['data']['variables']['xy_cat']
        self.xy_real= config['data']['variables']['xy_real']
        self.t_real= config['data']['variables']['t_real']
        self.labels_t= config['data']['variables']['labels_t']
        self.predict_features= config['data']['variables']['predict_features']
        self.use_txy_mask_as_input= config['data']['variables']['use_txy_mask_as_input']
        self.return_labels_t= config['data']['compute']['return_labels_t']
        self.pred_period= config['data'].get('prediction_period', 1) #Number of timesteps to predict into the future
        #self._real= [] #['lon', 'lat'] #Actually, this would be xy_real, because we can build a static map
        
        #Get minmax standarization values for _real data
        self.txy_real_minmax= config['data']['minmax']['txy_real_minmax']
        self.xy_real_minmax= config['data']['minmax']['xy_real_minmax'] #This is the dem
        self.t_real_minmax= config['data']['minmax']['t_real_minmax']
        
        #Get class meanings for _cat and _mask classes
        self.txy_cat_classes= config['data']['classes']['txy_cat']
        self.xy_cat_classes= config['data']['classes']['xy_cat']
        self.txy_mask_classes= config['data']['classes']['txy_mask']
        
        #Process _cat according to one of three possibilities. Assuming _cat has classes 0,1,2,3,4
        # - [1,3]: Set to one if class is 1 or 3
        # - [None]: Convert all classes to one hot and remove first dimension, similar to [[1,2,3,4]] (notice double list)
        # - [[1,2], [3,4]]: Get a one hot array with first entry for classes 1 or 2, and second entry for classes 3 or 4 
        self.txy_cat_process= config['data']['classes_process']['txy_cat']
        self.xy_cat_process= config['data']['classes_process']['xy_cat']
        self.txy_mask_process= config['data']['classes_process']['txy_mask']
        
        #Get other configuration parameters: custom processing of our data
        self.ignore_first_N= config['data']['compute'].get('ignore_first_N', 0)
        self.compute_ndvi= config['data']['compute']['ndvi']
        self.compute_clima= config['data']['compute'].get('clima', False)
        self.compute_clima_next= config['data']['compute'].get('clima_next', False)
        if self.compute_clima_next: assert self.compute_clima
        self.compute_t_next= config['data']['compute'].get('t_next', False)
        if self.compute_t_next: 
            print(f'Warning: Using {self.compute_t_next=} implies non-causal ERA5 usage, with the assumption that'
                  f'forecasts would be available for up to {self.pred_period} timesteps in the future')
        self.compute_era5_clima= config['data']['compute']['era5_clima']
        self.compute_era5_clima_next= self.compute_clima_next #config['data']['compute'].get('era5_clima_next', False)
        self.compute_era5_clima_delete_mean= config['data']['compute'].get('era5_clima_delete_mean', False)
        self.compute_era5_clima_delete_min_max_clima= config['data']['compute'].get('era5_clima_delete_min_max_clima', True)
        self.compute_era5_clima_detrend_min_max= config['data']['compute'].get('era5_clima_detrend_min_max', False)
        self.compute_smooth_clima= config['data']['compute'].get('smooth_clima', 1)
        self.compute_detrend_txy= config['data']['compute'].get('detrend_txy', False)
        if self.compute_detrend_txy: assert self.compute_clima, 'Clima computation is needed to apply detrending'
        self.compute_clima_before_and_after= config['data']['compute'].get('clima_before_and_after', False)
        
        #Process xy_reals as mean + dev
        self.compute_xy_real_mean_dev= config['data']['compute']['xy_real_mean_dev']

        #Final configs
        self.return_as_pytorch= config['data']['compute']['return_as_pytorch'] #If true, data is returned as pytorch
        self.return_just_cubes= config['data']['compute']['return_just_cubes'] #If true, all is concatenated into a single cube
        self.low_res= config['data']['compute']['low_res'] #If true, downscale image to half resolution (e.g.: 1/4 VRAM usage)
        self.half_precission= config['implementation']['trainer']['precission'] == 16
        self.sanitization_limits= config['data']['compute']['sanitization_limits']
        
        #Augment
        self.rng= np.random.default_rng(seed=None) #This is better for augmenting, if dataloaders are reset every time
        self.augment_rotate90= config['data']['augment']['rotate90']= True
        self.augment_flip= config['data']['augment']['flip']= True
        
        #Compute final feature names: this gets quite complex as it depends on all the preprocessing choices that were made
        self.data_names= { 'txy_real':None, 'txy_mask':None, 'txy_cat':None, #txy
                           'xy_cat':None, 'xy_real':None, #xy
                           't_real':None,} #t 
        txy_real_ndvi_names= self.txy_real + ([f'{NDVI}'] if self.compute_ndvi else []) 
        txy_real_ndvi_clima_names= [f'{v}_clima{"_next" if self.compute_clima_next else ""}' 
                                    for v in txy_real_ndvi_names] if self.compute_clima else []
        self.data_names['txy_real']= txy_real_ndvi_names +\
                        ([f'{n}_detrended' for n in txy_real_ndvi_names] if self.compute_detrend_txy else []) +\
                        txy_real_ndvi_clima_names +\
                        ([f'{n}_before' for n in txy_real_ndvi_clima_names] if self.compute_clima_before_and_after else []) +\
                        ([f'{n}_after' for n in txy_real_ndvi_clima_names] if self.compute_clima_before_and_after else [])
        self.data_names['txy_cat']= [f'{var}:{c}' for i, (var, pr) in enumerate(zip(self.txy_cat, self.txy_cat_process))
                                     for c in (list(self.txy_cat_classes[i].values())[1:] if pr is None
                                               else ['_OR_'.join(self.txy_cat_classes[i][i2] for i2 in pri) for pri in pr] 
                                                   if isinstance(pr[0], list)
                                               else ['_OR_'.join(self.txy_cat_classes[i][i2] for i2 in pr)])]
        self.data_names['txy_mask']= [f'{var}:{c}' for i, (var, pr) in enumerate(zip(self.txy_mask, self.txy_mask_process))
                                      for c in (list(self.txy_mask_classes[i].values())[1:] if pr is None
                                                else ['_OR_'.join(self.txy_mask_classes[i][i2] for i2 in pri) for pri in pr] 
                                                    if isinstance(pr[0], list)
                                                else ['_OR_'.join(self.txy_mask_classes[i][i2] for i2 in pr)])]
        self.data_names['xy_cat']= [f'{var}:{c}' for i, (var, pr) in enumerate(zip(self.xy_cat, self.xy_cat_process)) 
                                    for c in (list(self.xy_cat_classes[i].values())[1:] if pr is None
                                              else ['_OR_'.join(self.xy_cat_classes[i][i2] for i2 in pri) for pri in pr] 
                                                  if isinstance(pr[0], list)
                                              else ['_OR_'.join(self.xy_cat_classes[i][i2] for i2 in pr)])]
        self.data_names['xy_real']= self.xy_real if not self.compute_xy_real_mean_dev \
                                       else [f'{name}_{mode}' for mode in ['mean', 'dev'] for name in self.xy_real]
        t_real_names= ([c for c in self.t_real if not c.endswith('_mean')] if self.compute_era5_clima_delete_mean else self.t_real)
        self.data_names['t_real']= [
            f'{t}{"_detrend" if self.compute_era5_clima_detrend_min_max else ""}{"_next" if self.compute_t_next else ""}' 
                                    for t in t_real_names] +\
            ([f'{c}_clima{"_next" if self.compute_era5_clima_next or self.compute_clima_next else ""}' for c in self.t_real 
              if c.endswith('_mean') or not self.compute_era5_clima_delete_min_max_clima] if self.compute_era5_clima else [])
        
        #Early checks (other checks will be done later online)
        assert len(self.txy_real_minmax) == len(self.data_names['txy_real']),\
            f'{len(self.txy_real_minmax)=} != {len(self.data_names["txy_real"])=}, '+\
            f'with {self.txy_real_minmax=} and {self.data_names["txy_real"]=}'
        assert len(self.t_real_minmax) == len(self.data_names['t_real']),\
            f'{len(self.t_real_minmax)=} != {len(self.data_names["t_real"])=}, '+\
            f'with {self.t_real_minmax=} and {self.data_names["t_real"]=}'
        
        #Print summary
        print(f'Found {len(self.registry)} minicubes with {subset=}, {versions=}, '\
              f'{self.test_start_date=}, {self.ignore_first_N=} from {self.datacubes_path}')
        
    def get_row(self, cid):
        'cid can either be an int, the location_id string, or the cube_id string (not preferred)'
        if isinstance(cid, str): 
            if cid in self.registry.index: return self.registry.loc[cid]
            elif cid in self.registry.mc_id.values: return self.registry[self.registry.mc_id==cid]
            else: raise AssertionError(f'{cid=} not found in the regsitry')
        elif isinstance(cid, int): return self.registry.iloc[cid]
        else: raise AssertionError(f'{cid=} with {type(cid)=} must be int or str')
        
    def __getitem__(self, cid):
        'Get training-ready data'
        #Load xarray
        cube_path= self.get_row(cid).path
        minicube= xr.open_dataset(self.datacubes_path / cube_path, engine='zarr')
        if self.debug: print(f'{cube_path=}')
        
        #Aggregate events according to common consensus in the project
        if self.return_labels_t: 
            minicube= aggregate_events(minicube, aggregate_codes_independently=True, new_cube=False)
        
        #Load all data from the xarray into a dictionary
        data= {'txy_real':None, 'txy_mask':None, 'txy_cat':None, #txy
               'xy_cat':None, 'xy_real':None, #xy
               't_real':None} #t
        
        #t_idx (the actual dates)
        t_idx= minicube.time.values
        if self.test_start_date is None:
            test_start_idx= 0
        else:
            test_start_idx= np.argmin(np.abs(np.datetime64(self.test_start_date)-t_idx))
        if self.debug: print(f'{t_idx[test_start_idx]=}, {self.test_start_date=}')
        
        #Compute climatology dates: for now, do it in monthly basis and use only data up to self.test_start_date
        if self.compute_clima or self.compute_era5_clima:
            t_idx_datetime= [t.astype('M8[D]').astype('O') for t in t_idx] #Convert to python's datetime
            t_split_all= np.array([t.month -1 for t in t_idx_datetime]) #Split in e.g. months
            t_split_all_dom= np.array([t.day for t in t_idx_datetime]) #Day of month
            #t_split_all_dom_norm in [0,1]
            t_split_all_dom_norm= np.array([(t.day-1) / (calendar.monthrange(t.year ,t.month)[1]-1) for t in t_idx_datetime])
            t_split_all_doy= np.array([t.timetuple().tm_yday for t in t_idx_datetime]) #Day of year
            #t_split_all_doy_norm in [0,1]
            t_split_all_doy_norm= np.array([(t.timetuple().tm_yday-1) / (365-1+ calendar.isleap(t.year)) for t in t_idx_datetime])
            t_unique= np.unique(t_split_all)
            if self.debug: print(f'{t_unique=}')
        
        #txy_real (bands + ndvi), assumed always present
        txy_real= minicube[self.txy_real].to_array("band").values
        bands, time, lat, lon= txy_real.shape
        s_lat, s_lon= (lat//self.low_res, lon//self.low_res) if self.low_res else (lat, lon) #Scaled lat lon
        if self.low_res: txy_real= _scale(txy_real, dims=(s_lat, s_lon), method='cubic')
        if self.debug: print(f'{txy_real.shape=}')
        
        #txy_mask (cloud mask), assumed always present and same dimensions as txy_real data
        try:
            txy_mask= minicube[self.txy_mask].to_array().values
        except Exception as e:
            print(f'Exception {e} when reading {self.txy_mask} from {cube_path}')
            txy_mask= np.zeros_like(txy_real).astype(int)
        data['txy_mask']= _process_mask(txy_mask, [list(c.keys()) for c in self.txy_mask_classes],
                                             self.txy_mask_process)[None]
        if self.low_res: 
            data['txy_mask']= _scale(data['txy_mask'].astype(np.uint8), dims=(s_lat, s_lon), method='nn').astype(bool)
        #Extend mask to also include nans and all-zeros
        nan_in_at_least_one_band= np.isnan(txy_real).any(axis=0, keepdims=True)
        all_zeros_in_at_least_one_band= ~np.any(txy_real, axis=(0,-1,-2), keepdims=True)
        data['txy_mask']= data['txy_mask'] | nan_in_at_least_one_band | all_zeros_in_at_least_one_band
        if data['txy_mask'].mean() == 1.: 
            print(f'Warning: Sentinel 2 data unavailable for cube {cube_path}')
        
        #Compute NDVI
        if self.compute_ndvi:
            nir= txy_real[[self.txy_real.index('B8A')]]
            red= txy_real[[self.txy_real.index('B04')]]
            ndvi= _ndvi(red, nir)
            txy_real= np.concatenate([txy_real, ndvi], axis=0)
                
        #Compute ALL climatology?
        if self.compute_clima: 
            #Array to append to inputs with a new channel containing the clima of the current month
            bands_all= bands + (1 if self.compute_ndvi else 0)
            all_curr_clima= np.zeros((bands_all, time, s_lat, s_lon))
            
            if self.compute_clima_before_and_after:
                all_curr_clima_before= np.zeros((bands_all, time, s_lat, s_lon))
                all_curr_clima_after= np.zeros((bands_all, time, s_lat, s_lon))
                co_months= int(self.compute_clima_before_and_after)
                clima_offset= co_months * 6 #How many timesteps to look back and forward

            #Compute climatology causally e.g.: 12 x s_lon x s_lat
            bins= len(t_unique)
            all_clima= np.zeros((bands_all, bins, s_lat, s_lon))
            all_clima_count= np.zeros((1, bins, s_lat, s_lon))

            #Start computing clima causally
            for i, (s, dom) in enumerate(zip(t_split_all, t_split_all_dom_norm)):
                #Update clima from timestep i for month s
                if not (self.compute_clima_next or self.compute_clima_before_and_after):
                    valid_mask= ~data['txy_mask'][0,i]
                    all_clima[:, s, valid_mask]+= txy_real[:, i, valid_mask]
                    all_clima_count[0, s]+= valid_mask.astype(float)
                #If self.compute_clima_next or self.compute_clima_before_and_after, we update the clima 
                #buffer only after self.pred_period or a month has passed, respectively, so that
                #when all_clima is offset, no data from the future will actually have been used
                #This is slightly suboptimal, since clima for i is not allowed to use the current sample
                else:
                    if self.compute_clima_before_and_after and i >= clima_offset and i > self.pred_period:
                        i_before= i - clima_offset
                    elif self.compute_clima_next and i > self.pred_period:
                        i_before= i - self.pred_period
                    else:
                        continue #Just use all_clima{"","_bafore","_after"}'s default intialization of zero
                    s_before= t_split_all[i_before]
                    valid_mask= ~data['txy_mask'][0,i_before]
                    all_clima[:, s_before, valid_mask]+= txy_real[:, i_before, valid_mask]
                    all_clima_count[0, s_before]+= valid_mask.astype(float)

                #Compute all_curr_clima interpolating between monthly values
                #It is computed causally even if self.compute_clima_next is True
                all_curr_clima[:, i]= interpolate_clima(
                    all_clima, all_clima_count, s, dom, bins, interp_order=self.compute_smooth_clima)
                if self.compute_clima_before_and_after:
                    all_curr_clima_before[:, i]= interpolate_clima(all_clima, all_clima_count, 
                        s-co_months if s-co_months>=0 else bins-s-co_months, dom, bins, interp_order=self.compute_smooth_clima)
                    all_curr_clima_after[:, i]= interpolate_clima(all_clima, all_clima_count, 
                        s+co_months if s+co_months<=bins-1 else s+co_months-bins, dom, bins, interp_order=self.compute_smooth_clima)
                
            #Detrend txy
            txy_real_detrend= [txy_real - all_curr_clima] if self.compute_detrend_txy else []
                
            #Add before and after
            if self.compute_clima_before_and_after:
                # N=6 #6 timesteps ~= one month
                # all_curr_clima_before= np.concatenate([all_curr_clima[:, :N], all_curr_clima[:, :-N]], axis=1) 
                # all_curr_clima_after= np.concatenate([all_curr_clima[:, N:], all_curr_clima[:, -73:-73+N]], axis=1) 
                all_curr_clima= np.concatenate([all_curr_clima, all_curr_clima_before, all_curr_clima_after], axis=0)

            #Compute clima next
            if self.compute_clima_next:
                all_curr_clima= [np.concatenate([t[:, self.pred_period:], t[:, -self.pred_period:]], axis=1) 
                                 for t in all_curr_clima]

            #Append the final feature arrays to txy_real
            txy_real= np.concatenate([txy_real] + txy_real_detrend + [all_curr_clima], axis=0)

            #Check that there were some values
            if all_clima_count.sum() == 0.:
                print(f'Warning: all clima is nan for all time and spatial values in {cube_path}')

            if self.debug: 
                fig, axes= plot_clima(all_clima, all_clima_count, t_unique, self.compute_ndvi)
                fig.savefig((self.datacubes_path / 'debug' / f'{cube_path.replace("/","--")}_all_clima.png'), 
                    dpi=300, bbox_inches='tight')
                    
        # except Exception as e: 
        #     import traceback 
        #     traceback.print_exc()
        #     breakpoint()

        data['txy_real']= _standarize(txy_real, self.txy_real_minmax)
        if self.debug: print('Processed txy_real')
        
        #txy_cat (SCL, for now ignored)
        if len(self.txy_cat):
            data['txy_cat']= _process_mask(minicube[self.txy_cat].to_array().values, 
                [list(c.keys()) for c in self.txy_cat_classes], self.txy_cat_process)
        
        #xy_cat (lccs class)
        data['xy_cat']= _process_mask(
            _auto_scale(self.xy_cat, minicube,  dims=(s_lat, s_lon), method='nn'), 
            [list(c.keys()) for c in self.xy_cat_classes], self.xy_cat_process)
        
        #xy_real (DEM)
        xy_real= _auto_scale(self.xy_real, minicube,  dims=(s_lat, s_lon), method='cubic')
        if self.compute_xy_real_mean_dev:
            xy_real_mean= np.nanmean(xy_real, axis=(1,2), keepdims=True) #Get cube mean (sometimes it has some nan values...)
            xy_real_dev= xy_real-xy_real_mean
            xy_real= np.concatenate([xy_real_dev, np.tile(xy_real_mean, (1, s_lat, s_lon))], axis=0) #Add dev wrt mean
        data['xy_real']= _standarize(xy_real, self.xy_real_minmax)
        
        #t_real (weather stuff)
        t_real= minicube[self.t_real].to_array().values
        endings= ['_mean', '_min', '_max']
        t_real_subsets= [[t for t in self.t_real if t.endswith(ending)] for ending in endings]
        t_real_subsets_idx= [[i for i,t in enumerate(self.t_real) if t.endswith(ending)] for ending in endings]
        t_real_all, t_real_climas= [], [] #Final lists
        
        #Compute climatology causally
        if self.compute_era5_clima:
            for e_i, (ending, t_real_subset, t_real_subset_idx) \
                    in enumerate(zip(endings, t_real_subsets, t_real_subsets_idx)):
                #Clima arrays
                bins= len(t_unique)
                era5_clima= np.zeros((len(t_real_subset), bins))
                era5_clima_count= np.zeros((1, bins))
                
                #Input array
                t_clima= np.zeros((len(t_real_subset), time))
                
                for i,(s, dom) in enumerate(zip(t_split_all, t_split_all_dom_norm)): 
                    #Update clima buffers
                    era5_clima[:,s]+= t_real[t_real_subset_idx, i]
                    era5_clima_count[:,s]+= 1
                    
                    #Interpolate using currently available information
                    t_clima[:,i]= interpolate_clima(
                        era5_clima, era5_clima_count, s, dom, bins, interp_order=self.compute_smooth_clima)

                #Subtract clima from instantaneous values and append clima
                if ending == '_mean':
                    if not self.compute_era5_clima_delete_mean: t_real_all.append(t_real[t_real_subset_idx]-t_clima)
                    t_real_climas.append(t_clima)
                else:
                    if not self.compute_era5_clima_delete_min_max_clima: t_real_climas.append(t_clima)
                    if self.compute_era5_clima_detrend_min_max: #Use minmax clima for detrending
                        t_real_all.append(t_real[t_real_subset_idx]-t_clima)
                    else: #Use mean clima for detrending
                        t_real_all.append(t_real[t_real_subset_idx]-t_real_climas[0])
                        
                #Offset all by self.pred_period 
                #IMPORTANT: ERA5 data is also being offset, with assumption that forecasts would be available
                if self.compute_era5_clima_next:
                    t_real_climas= [np.concatenate([t[:, self.pred_period:], t[:, -self.pred_period:]], axis=1) 
                                    for t in t_real_climas]
                if self.compute_t_next:
                    t_real_all= [np.concatenate([t[:, self.pred_period:], t[:, -self.pred_period:]], axis=1) 
                                 for t in t_real_all]

        # except Exception as e: 
        #     import traceback 
        #     traceback.print_exc()
        #     breakpoint()
        data['t_real']= _standarize(np.concatenate(t_real_all + t_real_climas, axis=0), self.t_real_minmax)
        if self.debug: print('Processed t_real')         
            
        #labels_t (extreme event labels)
        if self.return_labels_t:
            #labels_t= labels_t.reshape(-1, 5).max(axis=1).reshape(1,-1)   #Does not work
            labels_t= minicube[self.labels_t].to_array().values.squeeze()
        else:
            labels_t= np.arange(time)*0 #Just a placeholder...
                  
        #Augment: we need to make a copy, because pytorch needs materialized arras, and some operations
        #just change the strides or do something else that does not create a new array
        if self.subset == 'train':
            #Rotate 0-270
            rot_times= self.rng.integers(0,4)
            if self.augment_rotate90 and rot_times > 0:
                for k,v in data.items():
                    if k.startswith('txy_') or k.startswith('xy_'): #(bands, time, lon, lat) or (bands, lon, lat)
                        if v is not None: data[k]= np.rot90(v, k=rot_times, axes=(-1, -2)).copy()

            #Flip vertically
            apply_flip= self.rng.integers(0,2) > 0.5
            if self.augment_flip and apply_flip:
                for k,v in data.items():
                    if k.startswith('txy_') or k.startswith('xy_'): #(bands, time, lon, lat) or (bands, lon, lat)
                        if v is not None: data[k]= np.flip(v, axis=-1).copy()
                        
        #Process nans: just set them to that band's mean value if it makes sense, or just to 0 otherwise
        #TODO: Better nan handling. e.g.: by holding last available value
        for k,v in data.items():
            if v is None: pass
            elif k.endswith('_real'):
                for band_idx in range(v.shape[0]): #Bands should always be in channel 0 after all previous processing
                    nans_in_band= np.isnan(v[band_idx])
                    if np.all(nans_in_band): 
                        print(f'Warning: Band index {band_idx} in data {k=} is nan for all spatio-temporal values in {cube_path}')
                        band_mean= 0.
                    else:
                        band_mean= np.mean(v[band_idx, ~nans_in_band])
                    data[k][band_idx, nans_in_band]= band_mean
            elif k.endswith('_mask') or k.endswith('_cat'):
                cat_nan= np.isnan(v)
                if cat_nan.any():
                    data[k][cat_nan]= 0
                    print(f'Warning: Missing categorical data in {cube_path}')
            else: pass #Ignore other data
            
        #Final shape assertions
        for k, arr in data.items():
            var_names= self.data_names[k]
            if arr is None: 
                assert self.data_names[k] == [] or self.data_names[k] is None,\
                    f'{k=} has {arr=}, but the associated {var_names=} are not empty or None'
            else:
                assert len(var_names) == arr.shape[0],\
                    f'{k}: {var_names=} has shape {len(var_names)} != {arr.shape[0]}'
        assert time == len(labels_t), f'{time=} != {len(labels_t)=}'
                
        #Keep only up to test_start_idx for the training set
        #This could be done more efficiently by avoiding reading the uneeded data from the beginning
        if self.subset == 'train' and test_start_idx > 0:
            for k,v in data.items():
                if k.startswith('txy_') or k.startswith('t_'): #(bands, time, lon, lat) or (bands, time)
                    if v is not None: data[k]= v[:, :test_start_idx]
            t_idx= t_idx[:test_start_idx]
            labels_t= labels_t[:test_start_idx]
       
        #Ignore some samples at the beginning?
        if self.ignore_first_N:
            for k,v in data.items():
                if k.startswith('txy_') or k.startswith('t_'): #(bands, time, lon, lat) or (bands, time)
                    if v is not None: data[k]= v[:, self.ignore_first_N:]
            t_idx= t_idx[self.ignore_first_N:]
            labels_t= labels_t[self.ignore_first_N:]
            if test_start_idx >= self.ignore_first_N: test_start_idx-= self.ignore_first_N
            
        #Save some debugging info
        self.last_cid, self.last_data, self.last_t_idx, self.last_minicube= cid, data, t_idx, minicube, 
        self.last_path, self.last_labels_t= cube_path, labels_t
        if self.debug: 
            print('Plotting')
            self.plot_last()
        
        #Rreturn dict with all the stuff
        return self._to_pytorch(data, t_idx, labels_t, cube_path, test_start_idx, pred_period=self.pred_period)
    
    def _to_pytorch(self, data, t_idx, labels_t, cid, test_start_idx, pred_period=1):
        #Everything to float
        data_final= {k:v.astype('float16' if self.half_precission 
                        else 'float32') for k,v in data.items() if v is not None}
        
        #Final sanitization, just in case
        for k, v in data_final.items():
            v[v<self.sanitization_limits[0]]= self.sanitization_limits[0]
            v[v>self.sanitization_limits[1]]= self.sanitization_limits[1]
        
        #Mix similar cubes together
        if self.return_just_cubes:
            assert 'txy_real' in data_final.keys()
            x, x_shape= data_final['txy_real'], data_final['txy_real'].shape
            data_names_x= []
            for k,v in data_final.items():
                if k == 'txy_real': #We already got this one in x, so we only need to update the data_names_x
                    data_names_x+= self.data_names[k]
                    continue
                elif k.startswith('txy_'): #(bands, time, lon, lat)
                    v_rep= v #No need to do work
                    if k == 'txy_mask' and not self.use_txy_mask_as_input: continue
                elif k.startswith('xy_'): #(bands, lon, lat)
                    v_rep= np.repeat(v[:, None], x_shape[1], axis=1)
                elif k.startswith('t_'): #(bands, time)
                    v_rep= np.repeat(v[:, :, None], x_shape[2], axis=2)
                    v_rep= np.repeat(v_rep[:, :, :, None], x_shape[3], axis=3)
                else:
                    raise AssertionError(f'Unknown kind of data dict index: {k}')
                x= np.concatenate([x, v_rep], axis=0) if x is not None else v
                data_names_x+= self.data_names[k]
                
            data_final= {'x':x[:,:-pred_period], 'labels':x[predict_idx,pred_period:], 'masks':data['txy_mask'][:,pred_period:]}
            data_names= {'x':data_names_x, 'labels':[f'{data_names_x[i]}_next' for i in predict_idx], 
                         'masks':[f'{name}_next' for name in self.data_names["txy_mask"]] }
            predict_idx= [data_names_x.index(band) for band in self.predict_features] #Bands to predict
            xy_label_metric_idx= [i for i,c in enumerate(data_names_x) if c.startswith('lccs_class:') and 
                   not any([c_bare in c for c_bare in ['no_data', 'urban', 'bare_areas', 'bare_areas_consolidated']]) ]
        else:      
            assert all([k in data_final.keys() for k in ['txy_mask', 'txy_real', 'xy_real', 't_real']])
            data_names= {'t':[], 'xy':[], 'txy':[], 'txy_mask':copy.copy(self.data_names['txy_mask'])}
            txy, data_names['txy']= data_final['txy_real'], copy.copy(self.data_names['txy_real'])
            xy, data_names['xy']= data_final['xy_real'], copy.copy(self.data_names['xy_real'])
            t, data_names['t']= data_final['t_real'], copy.copy(self.data_names['t_real'])
            for k,v in data_final.items():
                if k.startswith('txy_'): #(bands, time, lon, lat)
                    if k == 'txy_mask' and not self.use_txy_mask_as_input or k == 'txy_real': continue
                    txy= np.concatenate([txy, v], axis=0) #Ready for concatenation
                    data_names['txy']+= self.data_names[k]
                elif k.startswith('xy_'): #(bands, lon, lat)
                    if k == 'xy_real': continue
                    xy= np.concatenate([xy, v], axis=0)
                    data_names['xy']+= self.data_names[k]
                elif k.startswith('t_'): #(bands, time)
                    if k == 't_real': continue
                    t= np.concatenate([t, v], axis=0)
                    data_names['t']+= self.data_names[k]
                else:
                    raise AssertionError(f'Unknown kind of data dict index: {k}')
            data_final= {'t':t, 'xy':xy, 'txy':txy, 'txy_mask':data['txy_mask']}
            predict_idx= [data_names['txy'].index(band) for band in self.predict_features] #Bands to predict
            xy_label_metric_idx= [i for i,c in enumerate(data_names['xy']) if c.startswith('lccs_class:') and 
                   not any([c_bare in c for c_bare in ['no_data', 'urban', 'bare_areas', 'bare_areas_consolidated']]) ]
        
        #To pytorch?
        if self.return_as_pytorch:
            data_final= {k:torch.from_numpy(v).type(torch.half if self.half_precission 
                            else torch.float) for k,v in data_final.items() if v is not None} #.to(self.device)
        
        #Add extra metainfo
        data_final['meta']= {} #We reserve this dict for low-cost meta information
        data_final['meta']['xy_label_metric_idx']= xy_label_metric_idx
        data_final['meta']['txy_label_idx']= predict_idx
        data_final['meta']['names']= data_names
        data_final['meta']['id']= cid
        data_final['meta']['subset']= self.subset
        data_final['meta']['test_start_idx']= test_start_idx
        
        #I have to pass the dates as a string, as other objects are not allowed in pytorch dataloaders
        t_idx_str= list(map(_npdate2str, t_idx))
        if self.return_just_cubes: 
            data_final['meta']['index_x']= t_idx_str[:-pred_period]
            data_final['meta']['index_labels']= t_idx_str[pred_period:]
            data_final['meta']['event_labels']= labels_t[pred_period:]
        else: 
            data_final['meta']['index']= t_idx_str
            data_final['meta']['event_labels']= labels_t
                            
        #Return everything
        return data_final
    
    def __len__(self):
        return len(self.registry)
    
    def plot_last(self, select=slice(265,265+12,1)):
        'Plot last generated data dict'
        
        setup_matplotlib()
        
        #Plot txy
        import copy
        base_channels= [np.array([2,1,0])] + [[i] for i in range(3, len(self.txy_real) + int(self.compute_ndvi))]
        base_labels= ['RGB', 'B8A'] + (['NDVI'] if self.compute_ndvi else [])
        base_offset= 4 + int(self.compute_ndvi)
        channels= copy.copy(base_channels)
        labels= copy.copy(base_labels)
        offset= copy.copy(base_offset)
        if self.compute_detrend_txy: 
            channels= channels + [(np.array(c))+offset for c in base_channels]
            labels= labels + [f'{l}_detrend' for l in labels]
            offset+= base_offset
        if self.compute_clima:
            channels= channels + [(np.array(c))+offset for c in base_channels]
            labels= labels + [f'{l}_clima{"_next" if self.compute_clima_next else ""}' for l in base_labels]
            offset+= base_offset
        if self.compute_clima_before_and_after:
            channels= channels + [(np.array(c))+offset for c in base_channels]
            labels= labels + [f'{l}_clima_before{"_next" if self.compute_clima_next else ""}' for l in base_labels]
            offset+= base_offset
            channels= channels + [(np.array(c))+offset for c in base_channels]
            labels= labels + [f'{l}_clima_after{"_next" if self.compute_clima_next else ""}' for l in base_labels]
            offset+= base_offset
        stuff= plot_txy(self.last_data['txy_real'], self.last_data['txy_mask'], self.last_t_idx, 
                        channels, labels, select=select, #title=f'{self.last_cid} ({select=})'
                                    )
        if self.debug: 
            if len(stuff)==2:
                stuff[0].savefig((self.datacubes_path / 'debug' / f'{self.last_path.replace("/","--")}_txy.png'), 
                                 dpi=300, bbox_inches='tight')
            else:
                from IPython.display import display
                from PIL import Image
                display(Image.fromarray(stuff, 'RGB'))
        
        #Plot xy: xy_cat & xy_real
        stuff= plot_xy(
                self.last_data['txy_real'][np.array([2,1,0])], #RGB
                self.last_data['xy_real']*self.xy_real_minmax[0], self.last_data['xy_cat'], self.last_t_idx, None,
                ['RGB'] + (['DEM dev', 'DEM mean'] if self.compute_xy_real_mean_dev else self.xy_real),
                classes=self.xy_cat_classes[0], select=select,
            #title=f'{self.last_cid} ({select=})')
                    )
        if self.debug: 
            if len(stuff)==2:
                stuff[0].savefig((self.datacubes_path / 'debug' / f'{self.last_path.replace("/","--")}_xy.png'), 
                                 dpi=300, bbox_inches='tight')
            else:
                from IPython.display import display
                from PIL import Image
                display(Image.fromarray(stuff, 'RGB'))
        
        #Plot t
        fig, axes= plot_t(self.data_names['t_real'], self.last_data['t_real'], self.last_t_idx, 
                          labels_t=self.last_labels_t, columns=4)
        if self.debug: fig.savefig((self.datacubes_path / 'debug' / f'{self.last_path.replace("/","--")}_t.png'), 
                                   dpi=300, bbox_inches='tight')