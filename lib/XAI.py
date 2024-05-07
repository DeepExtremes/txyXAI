#General impots
from typing import List, Tuple, Optional, Union, Dict
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from pathlib import Path
import copy
import sklearn.cluster as sk_cluster, sklearn.decomposition as sk_decomposition, sklearn.manifold as sk_manifold
from sklearn.metrics import silhouette_score

#Custom imports
from .XAI_plot import (plot_attribution, plot_attributions_nd,
                      plot_explained_variance, plot_scatter_with_color)
from .XAI_utils import get_agg_mask, aggregate_inputs, event_at_positon, clip_outliers
from .XAI_attribute import attribute, XAIModelWrapper
from .XAI_attribute import XAIModelWrapper, attribute
from .data_utils import setup_matplotlib
setup_matplotlib(); #Create custom colormap, set default styles

eps = 1e-7 

#Main XAI class
class XAI():
    '''
    Class for handling eXplainable AI (XAI) experiments. It explains the predictions of the deep learning
    models by using any attribution method from `captum.attr` such as `Saliency`, `InputXGradient`, or `IntegratedGradients`.
    It should work for all kinds of output data (categorical / regresion) and all kinds of input / output
    dimensionality. However, only plots for some input / output dimensionality combinations are currently avaialable.
    In any case, a general plot showing average overall attributions of input features vs output features should always be generated.
    It also supports aggregation of the inputs (e.g. by mean) over specific dimensions, and aggregation of the outputs
    based on a mask (e.g.: consider only predictions that have low error, consider only predicitions given a certain ground
    truth class, or based on a custom compartmentalization).
    
    :param config: Global configuration dictionary for the case, containig all hyperparameters. Here we use:
        config['debug']: if >= 1, it runs the XAI only for config['debug'] samples and prints some extra info
        config['task']: 'Classification' or 'ImpactAssessment' (regression)
        config['data']['num_classes']: Number of output classes or features that have been predicted
        config['data']['input_features_flat']: All feature names in order in a flat list (not a list of lists)
        config['save_path']: XAI plots will be saved in config['save_path'] / 'xai'
        config['xai']['out_agg_dim']: None or tuple, if tuple, output dims wrt which we aggregate the output
        config['xai']['mask']: Used for masked aggregation, it selects the aggregation mode. 
            It must be one of ["none", "events[-full]", "correctness[-full]", "labels[-full]", "custom[-full]"]
        config['xai']['type']: Captum attribution method (str)
        config['xai']['outlier_perc']: Percentage of outliers to eliminate from the attributions (0-100) or None
        config['xai']['in_channels_rgb_2d']: RGB bands for plotting and attribution averaging
        config['xai']['out_channels_rgb_2d']: RGB bands for plotting and attribution averaging
        config['xai']['in_channels_rgb_3d']: RGB bands for plotting and attribution averaging
        config['xai']['out_channels_rgb_3d']: RGB bands for plotting and attribution averaging
        config['xai']['plots']: Choose what plots to compute, list, default: ['nd', '1d', '2d', '3d', 'avg', 'dim-reduction', 'cluster']
        config['xai']['global_downsampling']: Perform (if >1) structured downsampling of the data for global attrs, default 4
        config['xai']['non_zero_standarization']: If true, rescales attribs. so that only non-zero inputs are considered 
        config['xai']['clustering']['method']: One of sklearn.cluster. E.g.: 'KMeans' 
        config['xai']['clustering']['params']: Method parameters. E.g. {n_clusters:5, init:'k-means++', n_init:'warn', max_iter:300}
        config['xai']['clustering']['plot_prototypes']: Number of prototypes to plot for every cluster
        config['xai']['dim-reduction']['method']: One of sklearn.decomposition (e.g. PCA) or sklearn.manifold (e.g. TSNE)
        config['xai']['dim-reduction']['params']: Method parameters
        config['xai']['ignore_classes']: List of classes to ignore for attribution.
        config['xai']['agg_inputs']: None or list of lists of axes of inputs to agg. with mean
        config['xai']['agg_inputs_function']: One of ['mean', 'min', 'max', 'minmax'], 
            with minmax being either min or max, whatever is bigger in absolute value.
        config['data']['data_dim'] & config['arch']['input_model_dim']: Used by `adapt_variables`
    :param model: Pytorch model
    :param dataloader: The dataloader used by the Pytorch model
    '''
    def __init__(self, config:dict, model:torch.nn.Module, dataloader:torch.utils.data.DataLoader):
        self.debug= config.get('debug', False)
        self.config = config
        self.model = model
        self.loader = dataloader
        self.task = config.get('task', 'regression')
        assert self.task in ['regression', 'classification'], \
            f'{self.task=} not in ["regression", "classification"]'
        
        #Create directory to save figures for visualization
        self.save_path = Path(config['save_path']) / 'xai'
        self.save_path.mkdir(exist_ok=True)
        
        #Get class names and feature names
        self.feature_names= config['data']['input_features_flat']
        self.class_names= config['data']['output_classes']
        self.num_classes= len(self.class_names)
        
        #Over which dims to aggregate?: int, tuple or None
        self.out_agg_dim= config['xai'].get('out_agg_dim', None)
        
        #Other options
        self.ignore_classes= config['xai'].get('ignore_classes', [])
        self.agg_inputs= config['xai'].get('agg_inputs', None)
        self.agg_inputs_function= config['xai'].get('agg_inputs_function', 'mean')
        
        #Use an aggregation mask? One of: ["none", "correctness[-full]", "labels[-full]", "custom"]
        #The class names change depending on the agg_mode
        self.agg_mode= config['xai'].get('mask', 'none')
        # self.out_agg_dim= (2,3,4); self.agg_mode= 'none'
        self.agg_classes= self._update_class_names() #Classes used for aggregation if agg is -full
        assert self.agg_classes is None or self.agg_classes[0] is None or \
               self.agg_classes is not None and self.agg_mode.endswith('-full') 
        
        #Define RGB bands for plotting and attribution averaging
        self.in_channels_rgb_2d= self.config['xai'].get('in_channels_rgb_2d', [])
        self.out_channels_rgb_2d= self.config['xai'].get('out_channels_rgb_2d', [])
        self.in_channels_rgb_3d= self.config['xai'].get('in_channels_rgb_3d', [])
        self.out_channels_rgb_3d= self.config['xai'].get('out_channels_rgb_3d', [])
        
        #Choose percentage of attributions that get clipped out (does not affect nd plots or global attrs.)
        self.outlier_perc= config['xai'].get('outiler_percentage', 1.)
        
        #Control what plots to produce (in case the data allows for them)
        self.plots= config['xai'].get('plots', 
                                                  ['nd', '1d', '2d', '3d', 'avg', 'dim-reduction', 'cluster'])
        if self.debug: print(f' > Producing {self.plots=}')
        
        #Perform (if >1) structured downsampling of the data for global attrs.
        self.global_ds= config['xai'].get('global_downsampling', 4)
        
        #If true, rescales attribs. so that only non-zero inputs are considered 
        self.non_zero_standarization= config['xai'].get('non_zero_standarization', False)
        
        #Keeps only a subset of input classes. It can be None or, for a model with 3 inputs something like:
        # [ ['var1', 'var2', 'var3'], None (i.e. keep all), ['var4', 'var5'] ]
        self.keep_input_features= config['xai'].get('keep_input_features', None)
        
        #Clustering
        if 'clustering' in config['xai'].keys():
            self.clustering_method= config['xai']['clustering'].get('method', 'KMeans')
            self.clustering_params=\
                self.config['xai']['clustering'].get('params', {'n_clusters':5})
            self.n_proto= config['xai']['clustering'].get('plot_prototypes', 1)
        else:
            assert 'cluster' not in self.plots, \
                f'If `cluster` is in {config["evaluation"]["xai"]["params"]=}, '\
                f'then {config["evaluation"]["xai"]["params"]["clustering"]} must be defined'
            
        #Dimensionality reduction
        if 'dim-reduction' in config['xai'].keys():
            self.dimred_method= config['xai']['dim-reduction'].get('method', 'PCA')
            self.dimred_params= self.config['xai']['dim-reduction'].get('params', {})
        else:
            assert 'dim-reduction' not in self.plots, \
                f'If `dim-reduction` is in {config["evaluation"]["xai"]["params"]=}, '\
                f'then {config["evaluation"]["xai"]["params"]["dim-reduction"]} must be defined'
            
    def xai(self, events:Optional[torch.Tensor]=None, data=None):
        '''
        Performs XAI over a dataloader, saving the plots of the attributions independently for each sample
        
        :param events: tensor with same shape as model outputs `y` indicating to what event that output corresponds
            (or 0 if it corresponds to no event). It canot be None if agg_mode == "events[-full]"
        '''
        #Reset matplotlib config, change defaults, and get default list of colors
        color_list= setup_matplotlib()
        
        #Create attributions or use precalculated attributions contained in `data`
        if data is None:
            attributions, inputs, labels, predictions, masks, x_shapes, y_shape= attribute(
                self.config, self.model, self.loader, self.num_classes, self.out_agg_dim, self.agg_mode, 
                events=events, ignore_classes=self.ignore_classes, debug=self.debug)
        else:
            attributions, inputs, labels, predictions, masks, x_shapes, y_shape= data

        #If self.agg_inputs is not None, aggregate some inputs!
        if self.agg_inputs is not None: 
            attributions_copy, inputs_copy= copy.copy(attributions), copy.copy(inputs)
            x_shapes_copy= copy.copy(x_shapes)
            attributions, inputs, x_shapes= aggregate_inputs(attributions, inputs, x_shapes, y_shape,
                                         agg_inputs=self.agg_inputs, agg_inputs_function=self.agg_inputs_function)
                    
        #Visualizaton of single event attributions ----------------------------------------------------
        #Visualize attributions: there are different possible visualizations depending on the dimensionality of data
        #We will also concatenate all attributions for doing a global aggregate analysis
        attributions_global, inputs_global= [[] for _ in x_shapes], [[] for _ in x_shapes]
        labels_global, predictions_global= [[] for _ in x_shapes], [[] for _ in x_shapes]
        masks_global= [[] for _ in x_shapes]
        feature_names_global, x_shapes_global= [None for _ in x_shapes], [None for _ in x_shapes]
        event_names= list(attributions[0].keys())
        xai_method= self.config['xai']['type']
        
        for event_i, event in enumerate(pbar:=tqdm(event_names, desc='Processing single events', leave=False)):
            #Build list of features
            if not isinstance(attributions, tuple) and not isinstance(attributions, list): 
                raise AssertionError('All inputs and attributions should be a tuple or list. This is likely a bug')
            elif event_i == 0: #Run only once
                #attributions= ({'event1':att_a, }, {'event1':attb_b})
                start_idx, feature_names= 0, []
                for a in attributions:
                    n_in_features= a[event].shape[-1]
                    feature_names.append(self.feature_names[start_idx:start_idx+n_in_features])
                    start_idx+= n_in_features
                      
            #Iterate over input tensors
            for i, (attribution_item, input_item, feature_names_item, x_shape)\
                    in enumerate(zip(attributions, inputs, feature_names, x_shapes)):
                #Keep only some features
                if self.keep_input_features is not None and self.keep_input_features[i] is not None:
                    features_idx= [feature_names_item.index(f) for f in self.keep_input_features[i]]
                    feature_names_item= self.keep_input_features[i]
                    x_shape[-1]= len(features_idx)
                        
                    #attributions: ([out x, out y, out t], out classes, [in x, in y, in t], in features)
                    attribution_event= attribution_item[event][...,features_idx]
                    #inputs: ([in x, in y, in t], in features)')
                    input_event= input_item[event][...,features_idx]
                else:
                    attribution_event= attribution_item[event]
                    input_event= input_item[event]
                
                #Save data for global analysis
                #Perform some structured downsampling on the input to reduce the amount of data
                if len(x_shape) not in [2,3,4] or len(y_shape) != 1: #Unsupported global downsampling :(
                    attributions_global[i].append(attribution_event)
                    inputs_global[i].append(input_event)
                elif len(x_shape) == 2: 
                    attributions_global[i].append(attribution_event[:,::self.global_ds])
                    inputs_global[i].append(input_event[::self.global_ds])
                elif len(x_shape) == 3: 
                    attributions_global[i].append(attribution_event[:,::self.global_ds,::self.global_ds])
                    inputs_global[i].append(input_event[::self.global_ds,::self.global_ds])
                elif len(x_shape) == 4: 
                    attributions_global[i].append(
                        attribution_event[:,::self.global_ds,::self.global_ds,::self.global_ds])
                    inputs_global[i].append(input_event[::self.global_ds,::self.global_ds,::self.global_ds])
                x_shapes_global[i]= list(inputs_global[i][0].shape)
                #Output data and labels are not processed otherwise
                labels_global[i].append(labels[event]); predictions_global[i].append(predictions[event])
                masks_global[i].append(masks[event])
                feature_names_global[i]= feature_names_item
                
                #We can skip this if doing only global processing
                if not any([p in ['nd', '1d', '2d', '3d'] for p in self.plots]): continue

                #Get data for further visualizations (not used for nd plot)
                a_plot, a_min, a_max= clip_outliers(attribution_event, outlier_perc=self.outlier_perc)
                t_plot, l_plot, p_plot, m_plot= input_event, labels[event], predictions[event], masks[event]
                pbar.set_description(f'Visualizing explanations [input {i+1}/{len(inputs)}] [{event=}]')   

                #Set a global descriptive name
                name= 'agg-'+self.agg_mode if self.agg_mode != 'none' else \
                         ('agg-'+str(self.out_agg_dim) if self.out_agg_dim is not None else 'no-agg')
                name= f'{name} [{xai_method}] [{a_min:.2E} - {a_max:.2E}]'
                if self.agg_inputs is not None: name= f'{name} [inputs_agg={self.agg_inputs_function}]'
                if len(attributions) > 1: name= f'{name} [input {i+1}] {event}'
                                
                #Plot the attribution
                plot_attribution(
                          a_plot, t_plot, l_plot, p_plot, m_plot, x_shape, y_shape, feature_names_item, name,  
                          color_list, class_names=self.class_names, is_global=False, 
                          plots=self.plots, save_path=self.save_path,
                          in_channels_rgb_2d=self.in_channels_rgb_2d, out_channels_rgb_2d=self.out_channels_rgb_2d, 
                          in_channels_rgb_3d=self.in_channels_rgb_3d, out_channels_rgb_3d=self.out_channels_rgb_3d,
                          task=self.task)

        #Visualizaton of global attributions -------------------------------------------------------------
        #Global attributions aggregation
        #feature_names_global, x_shapes_global #Nothing to do here
        n_events= len(attributions_global[0])
        print(' > Stacking global arrays')
        #attributions: ([out x, out y, out t], out classes, [in x, in y, in t], in features)
        attributions_global= [np.stack(arr, axis=0) for arr in 
                              tqdm(attributions_global, desc=f'Processing attributions', leave=False)]
        #inputs: ([in x, in y, in t], in features)')
        inputs_global= [np.stack(arr, axis=0) for arr in 
                        tqdm(inputs_global, desc=f'Processing inputs', leave=False)]
        #labels: ([out x, out y, out t], [out classes]) | predictions ([out x, out y, out t], out classes)
        labels_global= [np.stack(arr, axis=0) for arr in 
                        tqdm(labels_global, desc=f'Processing labels', leave=False)]
        predictions_global= [np.stack(arr, axis=0) for arr in 
                             tqdm(predictions_global, desc=f'Processing predictions', leave=False)]
        #masks ([out x, out y, out t], out classes)
        masks_global= [np.stack(arr, axis=0) for arr in 
                       tqdm(masks_global, desc=f'Processing masks', leave=False)]
                
        #Compute average attributions ----------------------------------------------------------------------
        print_non_zero_std= self.non_zero_standarization and False #Do not print
        if 'avg' in self.plots:
            for i, (attribution_item, input_item, feature_names_item, x_shape) in \
                    enumerate(pbar:=tqdm(zip(attributions_global, inputs_global, 
                                             feature_names_global, x_shapes_global), total=len(inputs))):
                #Set a global descriptive name
                name= 'agg-'+self.agg_mode if self.agg_mode != 'none' else \
                         ('agg-'+str(self.out_agg_dim) if self.out_agg_dim is not None else 'no-agg')
                name= f'Global avg {name} [{xai_method}]'
                if self.agg_inputs is not None: name= f'{name} [inputs_agg={self.agg_inputs_function}]'
                if self.non_zero_standarization: name= f'{name} [non-zer-std]'
                name= f'{name} [{n_events=}] [{event_names[0]=}] [input {i+1}]'

                #Apply non-zero standarization?    
                if self.non_zero_standarization:
                    factor= input_item.reshape(-1, input_item.shape[-1]) #N x in_features
                    factor= factor.shape[0] / np.maximum((factor != 0).sum(axis=0), 1.) #in_features
                    attribution_item*=factor.reshape(*([1 for _ in attribution_item.shape[:-1]] + [factor.shape[0]]))

                #Average attributions
                use_paper_style= True #The non-default style I used for the paper
                #Visualize attributions for classes vs features aggregated over any amount of extra dimensions
                pbar.set_description(f'Computing global average attributions [input {i+1}/{len(inputs)}]')
                figsize_nd=(min(max(9, 17*len(feature_names_item)/15), 25), 9) #(17,9)
                fig, ax= plot_attributions_nd(
                              np.swapaxes(attribution_item, 0, len(y_shape)), [n_events] + x_shape, list(y_shape), 
                              feature_names_item if not print_non_zero_std else [f'{f} ({100/k:.2f}% non-zero)' 
                              for f,k in zip(feature_names_item, factor)], self.class_names, 
                              figsize= (6,6) if use_paper_style else figsize_nd,
                              orientation='h' if use_paper_style else 'v', 
                              plot_first_N=9 if use_paper_style else 100, 
                              textwrap_width=17 if use_paper_style else None)
                fig.savefig(self.save_path / f'{name}_nd.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                
        #If doing clustering or dim-reduction, flatten and merge together all attribution dimensions
        if any([p in ['cluster', 'dim-reduction'] for p in self.plots]):
            attr_all= np.concatenate([a.reshape(a.shape[0],-1) for a in attributions_global], axis=1)
                
        #Compute dimensionality reduction of attributions --------------------------------------------------------
        if 'dim-reduction' in self.plots:
            #Apply method
            print(f' > Using {self.dimred_method=} with parameters {self.dimred_params=}')
            try: dr_method= getattr(sk_decomposition, self.dimred_method)(**self.dimred_params)
            except: dr_method= getattr(sk_manifold, self.dimred_method)(**self.dimred_params)
            attr_transformed= dr_method.fit_transform(attr_all)
            
            #Set a global descriptive name
            name= 'agg-'+self.agg_mode if self.agg_mode != 'none' else \
                     ('agg-'+str(self.out_agg_dim) if self.out_agg_dim is not None else 'no-agg')
            if self.agg_inputs is not None: name= f'{name} [inputs_agg={self.agg_inputs_function}]'
            name= f'Global dim-reduction [{self.dimred_method}] {name} [{xai_method}] [{n_events=}]'
            
            #If PCA, do some custom stuff
            if self.dimred_method == 'PCA':
                #Plot cummulative explained variance
                plot_explained_variance(dr_method.explained_variance_ratio_, name, save_path=self.save_path);
                print(f' > PCA: Explained variance by first two components: '
                      f'{np.sum(dr_method.explained_variance_ratio_[:2])*100:.2f}%')
                
            #Do plots of first 2 dimensions with an input dimension as color (or a miniature of the RGB cube)
            for i, (attribution_item, input_item, feature_names_item, x_shape) in \
                    enumerate(pbar:=tqdm(zip(attributions_global, inputs_global, feature_names_global, 
                                             x_shapes_global), total=len(inputs), desc='Plotting PCA')):
                
                #Plot scatter of first 2 components vs features   
                plot_scatter_with_color(attr_transformed[:,[0]], attr_transformed[:,[1]], x_shape, input_item,
                    feature_names_item, columns=4, name=f'{name} [input {i+1}]', save_path=self.save_path)
                
                    
        #Compute clustering of attributions ----------------------------------------------------------------------
        #Plot some prototype attributions for each cluster, and compute average attribution within the cluster
        if 'cluster' in self.plots:
            #Do the clustering
            print(f' > {self.clustering_method=} with {self.clustering_params=} and {self.n_proto=} prototypes')
            cl_method= getattr(sk_cluster, self.clustering_method)(**self.clustering_params)
            #fit_transform gives us the distances from each sample to the centroid
            distances= cl_method.fit_transform(attr_all) 
            silhouette_sc= silhouette_score(attr_all, cl_method.labels_)
            print(f' > Found labels {cl_method.labels_=} for samples {event_names[:5]=} with {silhouette_sc=:.4f}')
            
            #Plot n prototypes from each cluster
            assert len(np.unique(cl_method.labels_)) == self.clustering_params['n_clusters']
            for cl_i in range(self.clustering_params['n_clusters']): #Iterate over number of clusters
                #Keep n_proto samples closest to the centroid
                closest_samples= np.argsort(distances[:,cl_i])[:self.n_proto] 
                for i, (attribution_item, input_item, feature_names_item, x_shape, attribution_item_global,
                        x_shape_global) in enumerate(zip(attributions_copy, inputs_copy, feature_names, 
                        x_shapes_copy, attributions_global, x_shapes_global)): #Iterate over inputs
                    
                    #Set a global descriptive name
                    name= 'agg-'+self.agg_mode if self.agg_mode != 'none' else \
                             ('agg-'+str(self.out_agg_dim) if self.out_agg_dim is not None else 'no-agg')
                    name= f'[{self.clustering_method}, class {cl_i} of '\
                          f'{list(range(self.clustering_params["n_clusters"]))}, '\
                          f'Global cluster [{silhouette_sc=:.4f}] {name} [{xai_method}]'
                    if self.agg_inputs is not None: name= f'{name} [inputs_agg={self.agg_inputs_function}]'
                    name= f'{name} [{n_events=}] [input {i+1}]'
                    
                    #Compute average attributions for those clusters
                    idx_curr_cluster= cl_method.labels_==cl_i #Select samples for current cluster
                    figsize_nd=(min(max(9, 17*len(feature_names_item)/15), 25), 9) #(17,9)
                    fig, ax= plot_attributions_nd(
                                      np.swapaxes(attribution_item_global[idx_curr_cluster], 0, len(y_shape)), 
                                      [sum(idx_curr_cluster)] + x_shape_global, list(y_shape), 
                                      feature_names_item if not print_non_zero_std else 
                                      [f'{f} ({100/k:.2f}% non-zero)' for f,k in zip(feature_names_item, factor)], 
                                      self.class_names, figsize=figsize_nd)
                    fig.savefig(self.save_path / f'{name}_nd.png', dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                    for cs_i, cs_v in enumerate(closest_samples): #Iterate over prototype idx
                        #Get event and name
                        event= event_names[cs_v]
                        name_plot= f'{name} {event} [Prototype {cs_i+1} of {len(closest_samples)}]'
                        
                        #Plot the attribution
                        a_plot, a_min, a_max= clip_outliers(attribution_item[event], outlier_perc=self.outlier_perc)
                        t_plot, l_plot, p_plot= input_item[event], labels[event], predictions[event]
                        m_plot= masks[event]
                        plot_attribution(
                          a_plot, t_plot, l_plot, p_plot, m_plot, x_shape, y_shape, 
                          feature_names_item, name_plot, color_list, self.class_names, is_global=True,
                          plots=self.plots, save_path=self.save_path,
                          in_channels_rgb_2d=self.in_channels_rgb_2d, out_channels_rgb_2d=self.out_channels_rgb_2d, 
                          in_channels_rgb_3d=self.in_channels_rgb_3d, out_channels_rgb_3d=self.out_channels_rgb_3d,
                          task=self.task)

        return (attributions_global, inputs_global, labels_global, predictions_global, masks_global, 
                feature_names_global, x_shapes_global, event_names)
            
    def _update_class_names(self):
        '''
        Updates `self.class_names` so that they correspond to the new classes after applying masked aggregation
        and removing ignore_classes
        '''
        agg_classes= None
        class_names_real= [c for i,c in enumerate(self.class_names) if i not in self.ignore_classes]
        if self.agg_mode == 'none': pass
        elif self.agg_mode == 'events-full':
            self.class_names= [f'{c1} (agg. by events: {c2})' for c1 in class_names_real for c2 in class_names_real]
        elif self.agg_mode == 'events':
            self.class_names= [f'{c1} (agg. by events: {c1})' for c1 in class_names_real]
        elif self.agg_mode == 'correctness-full':
        #E.g.: ['None (correct)', 'None (incorrect)', 'Convective storm (correct)', 'Convective storm (incorrect)']
            self.class_names= [f'{c1} (agg. by {c2} predictions)' 
                               for c1 in class_names_real for c2 in ['correct', 'incorrect']]
        elif self.agg_mode == 'correctness':
            #E.g.: ['None (correct)', 'Convective storm (correct)']
            self.class_names= [f'{c1} (agg. by {c2} predictions)' for c1 in class_names_real for c2 in ['correct']]
        elif self.agg_mode == 'labels':
            self.class_names= [f'{c1} (agg. by true label: {c1})' for c1 in class_names_real]
        elif self.agg_mode == 'labels-full':
            self.class_names= [f'{c1} (agg. by true label: {c2})' 
                               for c1 in class_names_real for c2 in class_names_real]
        elif self.agg_mode == 'custom' or self.agg_mode == 'custom-full':
            assert 'agg_classes' in self.config['xai'].keys(),\
                f"config['xai']['agg_classes'] must be a list with class names when "\
                f"using {self.agg_mode=} and a custom aggregation mask is provided"
            agg_classes= self.config['xai']['agg_classes']
            if self.agg_mode == 'custom':
                assert len(class_names_real) == len(agg_classes),\
                  f'If using {self.agg_mode=}, {class_names_real=} and {agg_classes=} should be the same length'
                self.class_names= [f'{c1} (agg. by: {c2})' if c2 is not None else c1 
                                   for c1,c2 in zip(class_names_real, agg_classes)]
            else:
                assert all([c2 is not None for c2 in agg_classes]),\
                    f'If using {self.agg_mode=}, {agg_classes=} cannot have None elements'
                self.class_names= [f'{c1} (agg. by: {c2})' for c1 in class_names_real for c2 in agg_classes]
        else:
            raise AssertionError(f'{self.agg_mode=} must be one of ["none", "events[-full]", "correctness[-full]",'\
                                 ' "labels[-full]", "custom[-full]"]')
        if self.debug: print(f' > Using {self.agg_mode=} with final {self.class_names=}')
        
        return agg_classes
        
# #Plot attributions of first 4 components
# for comp in [0,1,2,3]:
#     figsize_nd=(min(max(9, 17*len(feature_names_item)/15), 25), 9) #(17,9)
#     #We must extract the inverse_transformed first two components, in which all attributions were concatenated
#     #To do so, we must know between which two positions in the attr_all array the attribution i is located
#     attr_start= 0 if i==0 else np.prod(attributions_global[i-1].shape[1:])
#     attr_end= attr_start + np.prod(attribution_item.shape[1:])
#     attr_comp= dr_method.inverse_transform(attr_transformed[:,[comp]]
#                                            )[attr_start:attr_end].reshape(attribution_item.shape[1:])
#     fig, ax= plot_attributions_nd(attr_comp, x_shape, y_shape, feature_names_item, class_names, figsize=figsize_nd)
#     fig.savefig(self.save_path / f'{name} [component {comp+1}] [input {i+1}] nd.png', dpi=300, bbox_inches='tight')
#     plt.close(fig)


#Preprocessing -------------------------------------------------------------------------------
#If ignore_classes is not empty, get rid of all output classes that need be ignored
# if len(self.ignore_classes):
#     attributions, labels, predictions, masks, y_shape, class_names= ignore_classes(
#                 attributions, labels, predictions, masks, y_shape, self.class_names, 
#                 self.agg_classes, self.ignore_classes, full=self.agg_classes is not None)
# else:
#     class_names= self.class_names                                                  
# assert y_shape[-1] == len(class_names), f'Number of output classes after applying masking {y_shape[-1]=} '\
#     f' != Number of {len(class_names)=} with {class_names=}. If using custom-full aggregation, check that'\
#     f'it is returning the right amount of classes, otherwise this might be a bug'
