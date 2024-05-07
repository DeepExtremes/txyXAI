import numpy as np, pandas as pd, torch, gc
from datetime import datetime
from abc import ABC, abstractmethod

def empty_cache():
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

class DeepExtremesXAILoader(ABC):
    '''
        Wrapper around DeepExtremes to do XAI over some cubes
        This dataloader must return data dict with keys 'x', 'masks', 'agg_mask' & 'labels'
    '''
    def __init__(self, dataset, config, cubes, timesteps, device, history=None, 
                 N=100, subset='all', event_duration=None, debug_on_error=True, keep_classes=None):
        self.event_id= None
        self.unique_lc_names= None
        self.full= '-full' in config['xai']['mask']
        self.num_classes= len(config['data']['output_classes'])
        self.cubes= cubes
        self.device= device
        self.history= history
        self.debug_on_error= debug_on_error
        
        if 'class' in self.registry.columns and keep_classes is not None:
            keep_rows= self.registry['class'].isin(keep_classes)
            self.registry= self.registry[keep_rows]
            print(f'Removed {keep_rows.sum()} / {len(keep_rows)} because class was not in {keep_classes=}')
        else:
            print(f'Cannot filter out classes not in {keep_classes=} because'+\
                  ' registry does not have class column')
            
        #Use random cubes
        if self.cubes in ['val_random', 'test_random']:
            assert not self.full
            print(f'Warning: using {subset=}')
            assert isinstance(timesteps, int)
            self.registry= self.registry.iloc[:N]
            #Generate 1 month periods to explain
            # self.timesteps= [slice(t0:=np.random.randint(250, 300+365//5),t0+6,None) 
            #                  for _ in range(len(self.registry))]
            self.timesteps= [slice(t0:=timesteps,t0+6,None) 
                             for _ in range(len(self.registry))]
                
        #Select cubes from val set so that all landcovers have a similar amount of samples
        elif self.cubes in ['val_landcover_random', 'test_landcover_random']: 
            print(f'Warning: using {subset=}')
            assert self.full
            assert isinstance(timesteps, int)
            # self.cube_lc_names= data["meta"]["names"]['xy']
            #list(self.registry['class'].unique()) 
            self.unique_lc_names= ['soil', 'grassland', 'broadtree', 
                                   'needletree', 'urban', 'mixedtree'] 
            #Get first N cubes for every landcover class
            lc_registry= [self.registry[self.registry['class']==lc].iloc[:N] 
                          for lc in self.unique_lc_names]
            self.registry= pd.concat(lc_registry, axis=0) 
            #Generate 1 month periods to explain
            # self.timesteps= [slice(t0:=np.random.randint(250, 320+365//5),t0+6,None) 
            #                  for _ in range(len(self.registry))]
            self.timesteps= [slice(t0:=timesteps,t0+6,None) 
                             for _ in range(len(self.registry))]
                
        #Use cubes associated with an event ID
        elif isinstance(self.cubes, int):
            assert subset == 'test'
            assert self.full
            assert isinstance(self.cubes, int), f'{self.cubes=} must be an int with the event id'
            assert event_duration is not None, 'Specify duration of the event with `event_duration`'
            self.event_id= self.cubes
            has_event= self.registry.events.apply(lambda e_list: str(self.cubes) in str(e_list))
            self.registry= self.registry[has_event]
            #We need to define a range, since this is not exact
            event_duration_range= np.arange(event_duration-3, event_duration+4)
            event_is_N_days= self.registry.events.apply(lambda events: any([e[0]==self.cubes and (
                datetime.strptime(e[2], '%Y-%m-%d') - datetime.strptime(e[1], '%Y-%m-%d')).days in 
                                                            event_duration_range for e in events]))
            print(f'{event_is_N_days.sum()=} {self.cubes=} and {event_duration_range=} found out of {N=}'+\
                  f' (will only keep {N=})')
            assert event_is_N_days.sum() > N, f'Reduce {N=}, there are not enough cubes with that event'
            self.registry= self.registry[event_is_N_days]
            self.registry= self.registry.iloc[:N]
            #print(self.registry.index)
            assert all([t in ['event', 'year-event', 'before', 'year-before', 'after', 'year-after']
                        for t in timesteps])
            self.timesteps= timesteps
            self.event_timesteps= np.round(event_duration/5) 
            
        #Use cube ids specified through `cubes` as a list and some specific timesteps
        elif isinstance(self.cubes, list): 
            if not self.full: 
                assert not isinstance(timesteps[0], list)
                assert len(self.cubes) == len(timesteps), f'{len(self.cubes)} != {len(timesteps)}'
            if self.full:
                assert isinstance(timesteps[0], list) or isinstance(timesteps[0], slice)
            
            # assert all([c in self.registry.mc_id for c in cubes]),\
            # assert not (set(self.cubes) - set(self.registry.mc_id)),\
            #     f'Some of the provided cube ids could not be found in the registry: '\
            #     f'{set(self.cubes) - set(self.registry.mc_id)}'
            # self.registry= self.registry[self.registry.mc_id.isin(cubes)]
            #Allow for repeated cubes
            self.registry= pd.concat([self.registry[(self.registry.mc_id == cube) | (self.registry.index == cube)] 
                                      for cube in self.cubes], axis=0)
            self.timesteps= timesteps
            
        else:
            raise AssertionError('')
        
        print(f'Explaining {len(self.registry)} cubes')
        
    def __len__(self):
        return len(self.registry)
        
    @abstractmethod
    def _get_data(self, cid):
        pass
    
    @abstractmethod
    def _postprocess_data(self, final_batch, timestep=None):
        pass
        
    def __getitem__(self, cid):
        if (isinstance(cid, int) and cid >= len(self)):
            raise StopIteration
        try:
            final_batch, labels, masks, cid_true, event_labels= self._get_data(cid)
            assert final_batch['x'][0].shape[0] == 1, 'Remember to add batch dimension'

            #Output stuff
            agg_shape= list(labels.shape)
            if self.unique_lc_names is not None:
                agg_shape[1]= len(self.unique_lc_names)
                agg_mask= torch.zeros(agg_shape)
            elif self.full:
                agg_shape[1]= len(self.timesteps)
                agg_mask= torch.zeros(agg_shape)
            else:
                agg_mask= torch.zeros(agg_shape)
            
            event_names= []
            for b in range(final_batch['x'][0].shape[0]): #Iterate over batch index
                if self.cubes in ['val_random']:
                    timestep= self.timesteps[cid] 
                    agg_mask[b,:,timestep]= 1
                elif self.cubes in ['val_landcover_random']:
                    lc_class= self.registry.iloc[cid]['class']
                    timestep= self.timesteps[cid] 
                    # cube_lc_classes_idx= [i for i,lc in enumerate(self.cube_lc_names) 
                    #                   if any([lc2 in lc for lc2 in self.unique_lc_names])]
                    #The whole cube has a single lc
                    agg_mask[b, self.unique_lc_names.index(lc_class), timestep]= 1
                elif isinstance(self.cubes, int):
                    #Search for event timesteps
                    timesteps= np.argwhere(event_labels==self.event_id)[:,0]
                    if len(timesteps) != self.event_timesteps:
                        f'Warning: {len(timesteps)=} ({timesteps=}) != {self.event_timesteps=}'
                        
                    if 'event' in self.timesteps:
                        agg_mask[b,self.timesteps.index('event'),timesteps[0]:timesteps[-1]+1]= 1
                    if 'year-event' in self.timesteps: 
                        agg_mask[b,self.timesteps.index('year-event'),
                                 timesteps[0]-365//5:timesteps[-1]-365//5+1]= 1
                    if 'year-before' in self.timesteps: 
                        agg_mask[b,self.timesteps.index('year-before'),
                                 timesteps[0]-len(timesteps)-365//5:timesteps[-1]-len(timesteps)-365//5+1]= 1
                    if 'year-after' in self.timesteps: 
                        agg_mask[b,self.timesteps.index('year-after'),
                                 timesteps[0]+len(timesteps)-365//5:timesteps[-1]+len(timesteps)-365//5+1]= 1
                    if 'before' in self.timesteps: 
                        agg_mask[b,self.timesteps.index('before'),
                                 timesteps[0]-len(timesteps):timesteps[-1]-len(timesteps)+1]= 1
                    if 'after' in self.timesteps: 
                        agg_mask[b,self.timesteps.index('after'),
                                 timesteps[0]+len(timesteps):timesteps[-1]+len(timesteps)+1]= 1
                    timestep= np.s_[timesteps[0]:timesteps[-1]+1]
                elif isinstance(self.cubes, list): 
                    if not self.full:
                        timestep= self.timesteps[cid]
                        agg_mask[b,:,timestep]= 1
                    else:
                        for i, timestep in enumerate(self.timesteps): 
                            agg_mask[b,i,timestep]= 1
                else:
                    raise AssertionError('')
                event_names.append(f'{cid_true}-{timestep=}')

            if agg_mask.sum()==0:
                print(f'{agg_mask.sum()=} should not be 0!')
                if self.debug_on_error: breakpoint()

            #Prepare final_batch
            final_batch['masks']= masks
            final_batch['labels']= labels
            final_batch['agg_mask']= agg_mask
            final_batch['event_name']= event_names
            # final_batch['meta']= batch['meta']
            
            #Post process data?
            final_batch= self._postprocess_data(final_batch, timestep=timestep)

            #Return batch
            return final_batch
        
        except Exception as e: 
            import traceback 
            traceback.print_exc()
            print(e)
            if self.debug_on_error: breakpoint()
            return {} #Return empty batch