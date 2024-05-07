from typing import List, Tuple, Optional, Union, Dict
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable

from functools import reduce
import cv2

#Custom
from .visualization import plot_maps
from .data_utils import setup_matplotlib, _get_plot_cmap, _get_plot_center, _get_plot_limits

#Changes some defaults to reproduce the plots in the paper
PAPER_STYLE= False

def plot_explained_variance(explained_variance_ratio, name, save_path=None):
    fig, ax= plt.subplots(1,1, figsize=(7,7))
    ax.plot(np.cumsum(explained_variance_ratio))
    ax.set_xlabel('Number of PCA components')
    ax.set_ylabel('Cumulative explained variance')
    if save_path is not None:
        fig.savefig(save_path / f'{name} explained variance.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig, ax

def plot_scatter_with_color(x1, x2, x_shape, input_item, feature_names_item, columns=4, name='', save_path=None):
    axis_for_mean= tuple([x+1 for x,_ in enumerate(x_shape[:-1])]) #Ignore first and last axis
    rows= len(feature_names_item) // columns
    rows= rows if len(feature_names_item) % columns == 0 else rows + 1
    fig, axes = plt.subplots(rows, columns, figsize=(columns*5, rows*3))
    for vi, v_name in enumerate(feature_names_item):
        y= input_item[...,vi].mean(axis=axis_for_mean) #Compute average feature vector for all inputs
        ax= axes[vi//columns, vi%columns] if len(axes.shape) == 2 else axes[vi]
        im= ax.scatter(x1, x2, c=y, s=20, cmap='plasma')
        #ax.set_xlabel(f'First {self.dimred_method} component'); ax.set_ylabel(f'Second {self.dimred_method} component')
        ax.set_title(v_name)
        divider= make_axes_locatable(ax)
        cax= divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path / f'{name} wrt features.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
def plot_attribution(a_plot, t_plot, l_plot, p_plot, m_plot, x_shape, y_shape, feature_names_item, name, color_list, 
                     class_names, is_global=False, plots=['nd' '1d', '2d', '3d'], save_path='./',
                     in_channels_rgb_2d=[], out_channels_rgb_2d=[], in_channels_rgb_3d=[], out_channels_rgb_3d=[],
                     task='regression'):
    '''
        Plot data + attributions, selecting the appropiate plot depending on the dimensionality of the data
    '''
    #Visualize attributions for classes vs features aggregated over any amount of extra dimensions
    if ('nd' in plots or is_global):
        figsize_nd= (min(max(9, 17*len(feature_names_item)/15), 25), 9) #(17,9)
        fig, ax= plot_attributions_nd(a_plot, x_shape, y_shape, feature_names_item, class_names,
                                      figsize= (9,9) if PAPER_STYLE else figsize_nd,
                                      orientation='h' if PAPER_STYLE else 'v', 
                                      plot_first_N=9 if PAPER_STYLE else 100, 
                                      textwrap_width=15 if PAPER_STYLE else None)
        fig.savefig(save_path / f'{name} nd.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    #Select only some timesteps around the considered mask
    if PAPER_STYLE: #Disable by default. It may fail to work depending on the dim of the data
        if m_plot is not None:
            if m_plot.mean() == 0.:
                selected_ts= l_plot.shape[2] - 20
            else:
                #selected_ts= (m_plot!=0).sum(axis=(0,1,3)).argmax(axis=0)
                selected_ts= np.argwhere((m_plot!=0).sum(axis=(0,1,3)))[:,0]
            offsets= (-30, 4) #(-9, 4)
            plot_ts= np.s_[max(0, np.min(selected_ts) + offsets[0]):
                                  min(l_plot.shape[2], np.max(selected_ts) + offsets[1])]
    else:
        plot_ts= np.s_[:]

    #Visualize attribution over 1 input dim (e.g. time) + features
    if len(x_shape) == 2 and len(y_shape) < 3 and ('1d' in plots or is_global):
        #p_plot must be exactly of shape (out t, out classes)
        if len(p_plot) > 2:
            #I'm not sure this will always work
            agg_axis= tuple(list(range(len(p_plot.shape)))[:-2])
            p_plot= p_plot.mean(axis=agg_axis)
            l_plot= l_plot.mean(axis=agg_axis)
            if m_plot is not None:
                m_plot= m_plot.mean(axis=agg_axis)

        #If y_shape == 2, we cannot plot the attributions for all output timesteps, 
        #so we select the timestep where the first true event is happening
        if len(y_shape) == 2:
            timestep= (l_plot!=0).argmax(axis=0).max()
            if m_plot is None: m_plot= np.zeros_like(p_plot) #We show this selection with the mask
            m_plot[timestep]= 1
            a_plot= a_plot[timestep]

        #If there are no timesteps in the predictions, just create fake timesteps and position the predicitons at the end
        if len(p_plot.shape) == 1:
            t=t_plot.shape[0]
            l_plot, p_plot= event_at_positon(l_plot, t, position='end'), event_at_positon(p_plot, t, position='end')
            if m_plot is None:
                m_plot= np.zeros_like(p_plot) #We show this selection with the mask
                m_plot[-1]= 1
            else:
                m_plot= event_at_positon(m_plot, t, position='end')
        
        #Heuristic: If predictions' t < input's t, just fill it with zeros from the left
        if p_plot.shape[0] < t_plot.shape[0]:
            co, ti, fi= a_plot.shape
            p_plot_ext= np.zeros((ti, co))
            m_plot_ext= np.zeros((ti, co))
            l_plot_ext= np.zeros((ti, co))
            p_plot_ext[-p_plot.shape[0]:]= p_plot
            m_plot_ext[-p_plot.shape[0]:]= m_plot
            l_plot_ext[-p_plot.shape[0]:]= l_plot
            l_plot, p_plot, m_plot= l_plot_ext, p_plot_ext, m_plot_ext

        #Compute automatic figsize
        #timestep=None #TODO
        figsize=(min(max(8, 15*x_shape[0]/100), 30), min(max(8, 15*x_shape[-1])/10, 20)) #(15,15)
        if PAPER_STYLE: figsize=(figsize[0]/2,figsize[1])
        plot_ts_1d= plot_ts
        fig, axes= plot_attributions_1d( 
            a_plot[:,plot_ts_1d], t_plot[plot_ts_1d], l_plot[plot_ts_1d], p_plot[plot_ts_1d], m_plot[plot_ts_1d], 
            feature_names_item, class_names,
            color_list=color_list,
            figsize=figsize, dpi=200, #outlier_perc=outlier_perc, 
            attr_factor=0.5, alpha=0.5, margin_perc=50., 
            kind='stacked' if task == 'classification' else 'sidebyside', attr_baseline='feature',
            names_position='top',
            is_classification=True if task == 'classification' else False,
            title=f'Attributions' + ('' if len(y_shape) == 1 else f', explaining output {timestep=}')
            )
        fig.savefig(save_path / f'{name} 1d.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    #Visualize attributions over 2 input dims (e.g. x & y) + features
    if len(x_shape) == 3 and ('2d' in plots or is_global):
        #a_plot.shape (5, 128, 128, 35); t_plot.shape (128, 128, 35); l_plot.shape (128, 128, 494, 5)
        xy_time= 0 #plot_ts.start
        plot_attributions_3d(a_plot[:,:,:,None], t_plot[:,:,None], l_plot[:,:,[xy_time]], 
                 p_plot[:,:,[xy_time]], m_plot[:,:,[xy_time]],
                 feature_names_item, class_names, save_path=save_path / f'{name} 2d.png',
                 title=f'Attributions', backend='numpy', #outlier_perc=outlier_perc, 
                 in_channels_rgb=in_channels_rgb_2d, out_channels_rgb=out_channels_rgb_2d,
                 transpose=True, xlabels=[''])

    #Visualize attributions over 3 input dims (e.g. x, y & t) + features
    if len(x_shape) == 4 and ('3d' in plots or is_global):
        if len(a_plot.shape) != 5: 
            print(f'Attributions can only have shape (co x y t c), but {a_plot.shape=}. '
                  f'Please aggregate over the spatial/temporal output dimensions (x y t)')
        else:
            #Heuristic: If predictions' t < input's t, just fill it with zeros from the left
            if p_plot.shape[-2] < t_plot.shape[-2]:
                co, xi, yi, ti, fi= a_plot.shape
                p_plot_ext= np.zeros((xi, yi, ti, co))
                m_plot_ext= np.zeros((xi, yi, ti, co))
                l_plot_ext= np.zeros((xi, yi, ti, co))
                p_plot_ext[:, :, -p_plot.shape[-2]:]= p_plot
                m_plot_ext[:, :, -p_plot.shape[-2]:]= m_plot
                l_plot_ext[:, :, -p_plot.shape[-2]:]= l_plot
                l_plot, p_plot, m_plot= l_plot_ext, p_plot_ext, m_plot_ext
                
            plot_attributions_3d(a_plot[:,:,:,plot_ts], t_plot[:,:,plot_ts], l_plot[:,:,plot_ts], 
                 p_plot[:,:,plot_ts], m_plot[:,:,plot_ts],
                 feature_names_item, class_names, save_path=save_path / f'{name} 3d.png',
                 title=f'Attributions', backend='numpy', #outlier_perc=outlier_perc, 
                 in_channels_rgb=in_channels_rgb_3d, out_channels_rgb=out_channels_rgb_3d
                            )

    if len(x_shape) > 4:
        print(f'Attribution plotting for {len(x_shape)=} is not supported (yet?)')

def get_multiindex_df(data:np.ndarray, rows:List[List[str]], columns:List[List[str]], 
                      row_names:Optional[List[str]]=None, column_names:Optional[List[str]]=None, 
                      default_name:str='values') -> pd.DataFrame:
    '''
    Builds a multiindex + multicolumn pandas dataframe. For instance, from a `data` matrix of
    shape (a, b, c, d), if len(rows) = 2, and len(columns) = 2, the output dataframe will have
    two column levels, two index levels, and a x b x c x d total rows. It is assumed that
    first rows are taken from `data`, and then columns are taken from the remining dimensions
    of `data`
    
    :param data: array with a shape that is consistent with the rows + columns provided
    :param rows: a list of lists that will be used to build a MultiIndex, optionally empty list.
        For instance, the list at position 0 will contain a label for each of the features of data
        in the 0th dimension.
    :param columns: a list of lists that will be used to build a MultiIndex, optionally empty list
        For instance, the list at position 0 will contain a label for each of the features of data
        in the 0th dimension.
    :param row_names: a list of row names for the final DataFrame, len(row_names) = len(rows)
    :param column_names: a list of column names for the final DataFrame, len(column_names) = len(column_names)
    :param default_name: the default name to use for row and columns if they are not provided
        
    :return: MultiIndex pd.DataFrame containing the data
    '''
    shape= data.shape
    if not rows:
        rows= [[default_name]]
        row_names=None
        shape= (1, *shape)
    if not columns:
        columns= [[default_name]]
        column_names=None
        shape= (*shape, 1)
    
    row_index = pd.MultiIndex.from_product(rows, names=row_names)
    col_index = pd.MultiIndex.from_product(columns, names=column_names)
    row_lens= [len(r) for r in rows]
    column_lens= [len(r) for r in columns]
    assert (*row_lens, *column_lens) == shape, f'{[row_lens, column_lens]=} != {shape=} '
    return pd.DataFrame(data.reshape(np.product(row_lens), np.product(column_lens)), 
                        index=row_index, columns=col_index)

def plot_attributions_nd(data:np.ndarray, x_shape:List[int], y_shape:List[int], 
                         feature_names:List[str], class_names:List[str], figsize:Tuple[float]=(17,9),
                         max_values:Optional[int]=1e6, orientation:str='v', plot_first_N:int=100, 
                         textwrap_width:Optional[int]=None,
                         ) -> Tuple[mpl.figure.Figure, plt.axis]:
    '''
        Plots avarage attributions over all dimensions except for output classes and input features
        It should work for any kind of model and input / output dimensionality
        
        :param data: attributions array with shape ([out x, out y, out t], out classes, [in x, in y, in t], in features)
        :param x_shape: List [[in x, in y, in t], in features]
        :param y_shape: List [[out x, out y, out t], out classes]
        :param feature_names: List with the names of the in features
        :param class_names: List with the names of the output classes
        :param figsize: figsize to pass to plt.subplots
        :param max_values: maximum number of attribution values to consider (to speed up plotting)
        :param orientation: plot orientation, either 'h' (default) or 'v'
        :param plot_first_N: int, limit the plot to the first N most relevant features
        :param textwrap_width: int, if not None, force-wrap the label's text if longer than textwrap_width chars
        
        :return: Matplotlib figure and ax
    '''
    #Perform line wraping
    if textwrap_width is not None:
        import textwrap
        #We replace _ by - so that textwrap attempts to break on those first
        feature_names= [textwrap.fill(f.replace('_','-'), width=textwrap_width).replace('-','_') for f in feature_names]
    
    #Build multiindex df
    row_names= [f'Output dim {i}' for i in range(len(y_shape)-1)] + ['Output class']
    row_names+= [f'Input dim {i}' for i in range(len(x_shape)-1)] + ['Input feature']
    rows= [list(range(y_i)) for y_i in y_shape[:-1]] + [class_names]
    rows+= [list(range(x_i)) for x_i in x_shape[:-1]] + [feature_names]
    attr_df= get_multiindex_df(data=data, rows=rows, columns= [], 
                               row_names=row_names, column_names= [], default_name='Attributions')

    #These transformations are needed for plotting
    if max_values is not None and len(attr_df) > int(max_values):
        attr_df= attr_df.sample(int(max_values))
    attr_df_ri= attr_df.reset_index()
    attr_df_ri.columns= attr_df_ri.columns.to_flat_index()
    attr_df_ri= attr_df_ri.rename({c:c[0] for c in attr_df_ri.columns}, axis=1)
    
    #Remove nans. Very important! This is mandatory for custom-full explanations to be correct
    attr_df_ri= attr_df_ri[~attr_df_ri['Attributions'].isna()]
    
    #Order by mae, filter out input features that only have zero-valued attribution
    feature_mae= {f:np.abs(attr_df_ri.loc[attr_df_ri['Input feature']==f, 'Attributions']).mean() for f in feature_names}
    sorted_input_features= [f for f in feature_names if feature_mae[f] > 0.]
    sorted_input_features.sort(key=lambda f: -feature_mae[f])
    
    #Get rid of lower features
    K= len(sorted_input_features)/len(feature_names)
    if plot_first_N < len(attr_df_ri):
        sorted_input_features= sorted_input_features[:plot_first_N]
        
    #Final filter
    attr_df_ri_filtered= attr_df_ri[attr_df_ri['Input feature'].isin(sorted_input_features)]
    
    #Plot and save
    if orientation in ('v', 'vertical'):
        figsize= (figsize[0]*K, figsize[1])
        x, y= 'Input feature', 'Attributions'
    else:
        x, y= 'Attributions', 'Input feature'
    fig, ax= plt.subplots(figsize=figsize)
    
    sns.barplot(attr_df_ri_filtered, ax=ax, x=x, y=y, hue='Output class',
                order=sorted_input_features, hue_order=class_names, orient=orientation)
    
    if orientation in ('v', 'vertical'):
        with warnings.catch_warnings():  #Raises warning in latest Matplotlib that we can ignore
            warnings.simplefilter("ignore")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
            sns.move_legend(ax, "upper right")
    else:
        sns.move_legend(ax, "lower right")
    #ax.legend([],[], frameon=False) #Remove legend?
    ax.grid(True)
    ax.set_title('Attribution averaged (with CI) over all other dimensions' + 
                 (f' (top {plot_first_N})' if plot_first_N < len(attr_df_ri) else ''))
    return fig, ax
    
def plot_attributions_1d(attributions:np.ndarray, #(out classes, in t, in features)
                         inputs:np.ndarray, #(in t, in features)
                         labels:np.ndarray, #(out t, [out classes]) out classes is optional
                         predictions:np.ndarray, #(out t, out classes)
                         masks:np.ndarray, #(out_t, out_classes)
                         feature_names:List[str], #List of in features names
                         class_names:List[str], #List of out classes names
                         plot_classes_predictions:Optional[List[str]]=None, 
                         #Classes to plot for predictions. If None, first class is ignored
                         plot_classes_attributions:Optional[List[str]]=None, 
                         #Classes to plot for attributions. If None, first class is ignored
                         figsize:Tuple[float]=(10,10), #Matplotlib figsize
                         #out t wrt which attributions are plotted. If None, use out t where first event occurs
                         color_list=list(mpl.colors.TABLEAU_COLORS),
                         title:Optional[str]=None,
                         margin_perc:float=25., #Add some % of margin to feature plots
                         alpha:float=0.8, 
                         attr_factor:float=0.5,
                         attr_baseline:str='middle', #One of {'middle', 'feature'}
                         kind:str='stacked', #One of {'stacked', 'sidebyside'}
                         names_position:str='left', #One of {'left', 'top'}
                         label_fontsize:float=11,
                         is_classification:bool=True,
                         **kwargs) -> Tuple[mpl.figure.Figure, plt.axis]:
    '''
    Plot 1D attributions (e.g. of a inputs) given an output timestep.
    If you want to see absolute attributions, pass np.abs(attributions) instead.
    
    :param attributions: array with shape (out classes, in t, in features)
    :param inputs: array with shape (in t, in features)
    :param labels: array with shape (out t, [out classes]) out classes is optional
    :param predictions: array with shape (out t, out classes)
    :param masks: array with shape (out t, out classes)
    :param feature_names: List of in features names
    :param class_names: List of out classes names
    :param plot_classes_predictions: Classes to plot for predictions. If None, first class is ignored 
        if there are more than 2 classes
    :param plot_classes_attributions: Classes to plot for attributions. If None, first class is ignored 
        if there are more than 2 classes
    :param figsize: Matplotlib figsize
    :param timesteps: Global x values
    :param timestep: output timestep with respect to which which attributions are plotted. 
        If None, use the first timestep where the ground truth output class != 0
    :param color_list: list of Matplotlib-compatible colors
    :param title: title of the plot
    :param margin_perc: Add some % of margin to feature plots
    :param alpha: transparency of the bars
    :param attr_factor: mutiply attribution values by this number (e.g., if 0.5, the attribution can only
        take maximum 50% of the height of the plot. Recommended: 0.5 for signed attributions, and 1
        for absolute attributions (this is done by default)
    :param attr_baseline: Where to place the attribution bars with respect to the plot.
        attr_baseline='middle': start attribution bars from the middle of the plot
        attr_baseline='feature': start attribution bars from the feature value for that timestep
    :param kind: How to plot attributions: 
        kind='stacked': plot attributions for each class stacked on top of each other
        kind='sidebyside': plot attributions for each class side by side
    :param names_position: Where to position the names of the features
        names_position='left': position them to the left of the plot
        names_position='top': position them to the top of the plot
    :param label_fontsize: font size of the labels
    :param is_classification: whether it is a classification or a regression problem
    
    :param *kwargs: kwargs to be passed to plt.subplots
    
    :return: Matplotlib figure and ax
    '''   
    #General checks
    #assert len(class_names) > 1, f'There must be at least 2 classes, found {len(class_names)=}'
    if is_classification and not np.all(labels.astype(int) == labels): 
        warnings.warn(f'The problem has been defined as {is_classification=}. Are you sure it is?')
    to= labels.shape[0] if len(labels.shape) > 1 else 1
    assert len(attributions.shape) == 3, f'{attributions.shape=} != 3 (out classes, in t, in features)'
    co, ti, fi= attributions.shape
    assert inputs.shape == (ti, fi), f'{inputs.shape=} != {(ti, fi)=} (in t, in features)'
    assert len(feature_names) == fi, f'{len(feature_names)=} != {fi=} (in features)'
    assert len(class_names) == co, f'{len(class_names)=} != {co=} (out classes)'
    if plot_classes_predictions is None: 
        if is_classification and co > 2: plot_classes_predictions= class_names[1:] #Ignore class 0 by default
        else: plot_classes_predictions= class_names
    assert set(plot_classes_predictions).issubset(set(class_names)),\
        f'{plot_classes_predictions=} must be in {class_names=}'
    if plot_classes_attributions is None: 
        if is_classification and co > 2: plot_classes_attributions= class_names[1:]
        else: plot_classes_attributions= class_names
    assert set(plot_classes_attributions).issubset(set(class_names)), \
        f'{plot_classes_attributions=} must be in {class_names=}'
    assert kind in ['stacked', 'sidebyside'], f'{kind=} must be in {["stacked", "sidebyside"]}'
    assert attr_baseline in ['middle', 'feature'], f'{attr_baseline=} must be in {["middle", "feature"]}'
    assert names_position in ['left', 'top'], f'{attr_baseline=} must be in {["left", "top"]}'
    assert len(color_list) >= len(plot_classes_predictions) and len(color_list) >= len(plot_classes_attributions),\
    f'Not enough colors in {len(color_list)=} for {len(plot_classes_predictions)=} or {len(plot_classes_attributions)=}'
    
    #Data processing
    timesteps= np.arange(0, ti)
    #timesteps_out = np.arange(0, to)
    cl_idx_predictions= np.array([class_names.index(c) for c in plot_classes_predictions])
    cl_idx_attributions= np.array([class_names.index(c) for c in plot_classes_attributions])
    
    #Set attributions to somewhere between 0 & 1
    is_attr_abs= not np.any(attributions < 0) #Attr are absolute if there is none below zero
    if is_attr_abs and attr_factor != 1.: 
        attr_factor= 1.
        warnings.warn('If attributions are absolute, attr_factor is automatically set to 1')
    attr= attributions
        
    #Build figure
    fig, axes = plt.subplots(figsize=figsize, nrows=fi + 3 if (np.array(masks)!=None).all() else fi + 2, 
                             sharex=True, **kwargs)
    axes= np.array([axes]) if not isinstance(axes, np.ndarray) else axes
    x_unit= timesteps[1]-timesteps[0] #Assume timesteps are equally spaced
    bar_width= 0.9 * x_unit if len(timesteps) > 1 else 0.9
    line_kwargs= dict(colors='gray', label='', linewidth=0.5, zorder=-1)
    
    #We define a function to plot a single feature + class on an mpl axis
    def plot_feature(ax:plt.axis, cl_idx:List[float], cl_names:List[str], absolute:bool=False, show_legend:bool=False,
                     ylabel:str='', x:Optional[np.ndarray]=None, y:Optional[np.ndarray]=None, y_is_attr:bool=False,
                     ylim:Tuple[float]=None):
        '''
        Plot on `ax` a scalar feature x and/or a category-like/attribution feature y
        
        :param ax: axis on which to plot
        :param cl_idx: array or list of class indices, to be used to index over y[:,i]
        :param cl_names: class names associated with `cl_idx`, referring to y
        :param absolute: whether the attribution data is absolute (>=0) or not
        :param show_legend: whether to plot the legend
        :param ylabel: label of the axis, i.e. name of the feature
        :param x: array of scalar features, with feature name associated `label`
        :param y: array of categorical features or attribution features, with classes / features
            associated to them in `cl_idx` and `cl_names`
        :param y_is_attr: wheter y is an attribution or a categorical feature (possibly with probabilities)
        :param ylim: y limits of the plot. If None, they are computed automatically from x.
            If None and x also None, they are set to (0,1)
        '''                   
        #Set limits
        if ylim is not None:
            ymin, ymax = ylim[0], ylim[1]
        elif x is not None:
            sp= margin_perc/100 * (np.max(x) - np.min(x))
            ymin, ymax= np.min(x) - sp, np.max(x) + sp
        elif not is_classification:
            sp= margin_perc/100 * (np.max(y) - np.min(y))
            ymin, ymax= np.min(y) - sp, np.max(y) + sp
        else:
            ymin, ymax= 0, 1
        if y_is_attr: y*= (ymax - ymin) * attr_factor
        
        #Plot categorical
        if y is not None: 
            #Decide where the categorical plot starts: either bottom, middle, or at the feature value
            if not y_is_attr:
                baseline= np.zeros(ti)
            elif absolute: #If using absolute attributions, always start from bottom
                baseline= np.zeros(ti) + ymin
            elif attr_baseline == 'feature': #Start from features if they exist, otherwise from the bottom
                baseline= x if x is not None else np.zeros(ti) + ymin
            elif attr_baseline == 'middle': #Start from the middle
                baseline= np.zeros(ti) + ymin + (ymax - ymin)/2
                ax.hlines(ymin + (ymax - ymin)/2, timesteps[0], timesteps[-1], **line_kwargs) 
            else: 
                raise AssertionError(f'Unknown {attr_baseline=}')
                 
            if kind == 'stacked':
                #There are two bottoms: one for the data going up, and one for the data going down
                bottom_up, bottom_down= np.copy(baseline), np.copy(baseline)
                for cl_i, cl_name in zip(cl_idx, cl_names):
                    going_up= y[:,cl_i] >= 0 #Boolean matrix idicating which data goes up now
                    bottom= np.where(going_up, bottom_up, bottom_down)
                    ax.bar(timesteps, y[:,cl_i], bar_width, label=cl_name, 
                           bottom=bottom, color=color_list[cl_i], alpha=alpha)
                    #Update either bottom_up or bottom_down accordingly
                    bottom_up[going_up]= bottom_up[going_up] + y[going_up, cl_i]
                    bottom_down[~going_up]= bottom_down[~going_up] + y[~going_up, cl_i]

            elif kind == 'sidebyside':
                class_bar_width= bar_width / len(cl_names)
                for i, (cl_i, cl_name) in enumerate(zip(cl_idx, cl_names)):
                    ax.bar(timesteps + i * class_bar_width, 
                           y[:,cl_i], class_bar_width, label=cl_name, 
                           bottom=baseline, color=color_list[cl_i], alpha=alpha)
            else:
                raise AssertionError(f'Unknown {kind=}')
            
        #Plot feature
        if x is not None:
            if not y_is_attr: #If it is not an attribution line, plot every class
                for cl_i, cl_name in zip(cl_idx, cl_names):
                    ax.plot(timesteps, x[:,cl_i], label=cl_name if y is None else None, 
                            color=ax.set_prop_cycle(color=color_list[cl_i]))
            else: #If it is an attribution line, it is just one feature, plot it
                ax.plot(timesteps, x, color=ax.set_prop_cycle(color=color_list[cl_idx[0]]))                
        
        #Set ax properties
        #if show_legend: ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.set_xlim([timesteps[0] - x_unit/2, timesteps[-1] + x_unit/2])
        if names_position == 'left': 
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
            ax.yaxis.set_label_coords(-0.1,0.5)
        elif names_position == 'top':
            ax.set_title(ylabel, fontsize=label_fontsize, pad=2)
        else:
            raise AssertionError(f'Unknown {names_position=}')
        if x is not None: ax.grid(True, zorder=-10, linestyle='dashed', linewidth=0.5)
      
    #Plot actual and predicted class
    use_bar_plot= predictions[:-1].sum() == 0 #Use bar plot if we extended the pred. with zeros
    plot_feature(axes[0], cl_idx_predictions, plot_classes_predictions,
                 absolute=True, show_legend=True, ylabel='GT', 
                 x=None if use_bar_plot else labels,
                 y=labels if use_bar_plot else None,
                 ylim=(0,1) if is_classification else None)
    plot_feature(axes[1], cl_idx_predictions, plot_classes_predictions,
                 absolute=True, show_legend=False, ylabel='Predicted', 
                 x=None if use_bar_plot else predictions,
                 y=predictions if use_bar_plot else None,
                 ylim=(0,1) if is_classification else None) 
    
    # Plot attribution mask
    if (np.array(masks)!=None).all():
        plot_feature(axes[2], cl_idx_predictions, plot_classes_predictions,
                     absolute=True, show_legend=False, ylabel='Attr. mask', 
                     y=masks, ylim=(0,1))     
    
    #Plot input features
    for f, ax in enumerate(axes[3:] if (np.array(masks)!=None).all() else axes[2:]):       
        plot_feature(ax, cl_idx_attributions, plot_classes_attributions,
                     absolute=is_attr_abs, show_legend=f==0, 
                     ylabel=feature_names[f], 
                     x=inputs[:,f], y=attr[...,f].T, y_is_attr=True)
        
    #Figure-wide configuration
    if PAPER_STYLE: axes[-1].set_xticks(timesteps[::(ti//10 if ti//10 else 1)])
    else: axes[-1].set_xticks(timesteps[::(ti//30 if ti//30 else 1)])
    title= 'Attributions' if title is None else title
    if not PAPER_STYLE: fig.suptitle(title, y=0.91, size=13)
    fig.subplots_adjust(hspace=0.4 if names_position == 'top' else 0.15)
    
    return fig, axes

def plot_attributions_3d(attributions:np.ndarray, inputs:np.ndarray, labels:np.ndarray, 
                         predictions:np.ndarray, masks:np.ndarray, 
                         feature_names:List[str], class_names:List[str], #outlier_perc:float=0.1,
                         save_path:Optional[Union[Path,str]]=None, title:str='', backend:str='numpy',
                         in_channels_rgb:List[str]=['B04', 'B03', 'B02'], 
                         out_channels_rgb:List[str]=['B04', 'B03', 'B02'],
                         xlabels=None, **kwargs):
    '''
    Plots attributions of 3D data alongside the data, i.e (x y t c) or (x y z c), and attributions (co x y t ci)
    
    :param attributions: array with shape (out classes, in x, in y, in t / in z, in features)
    :param inputs: array with shape (in x, in y, in t / in z, in features)
    :param labels: array with shape (out x, out y, out t / out z, out classes)
    :param predictions: array with shape (out x, out y, out t / out z, out classes)
    :param masks: binary array with shape (out x, out y, out t / out z, out classes)
    :param feature_names: List of in features names
    :param class_names: List of out classes names
    :param save_path: Path of the output image, or None to not save anything
    :param title: Title of the figure (to be ignored if backend is numpy)
    :param backend: `numpy` or `matplotlib`. See more info in `plot_maps`
    :param in_channels_rgb: List of three input features corresponding to RGB channels. They will be aggregated and plotted together
    :param out_channels_rgb: List of three output classes corresponding to RGB channels. They will be aggregated and plotted together
        Also, the corresponding attributions will be averaged over the RGB channels for plotting.
    :param xlabels: labels for the xaxis, or None to get just indices
        
    :return: Matplotlib figure and ax if backend == `matplotlib` or RGB image array, if backend == `numpy`
    '''
    reduce_lists= lambda lists: reduce(lambda x, y: x + y, lists, [])
    
    #General shape assertions
    assert len(attributions.shape) == 5, \
        f'{attributions.shape=} != 5 (out classes, in x, in y, in t / in z, in features)'
    co, xi, yi, ti, fi= attributions.shape
    assert inputs.shape == (xi, yi, ti, fi), f'{inputs.shape=} != ({xi=}, {y1=}, {ti=}, {fi=})'
    assert predictions.shape == (xi, yi, ti, co), f'{predictions.shape=} != ({xi=}, {yi=}, {ti=}, {co=})'
    assert labels.shape == (xi, yi, ti, co), f'{labels.shape=} != ({xi=}, {yi=}, {ti=}, {co=})'
    assert masks.shape == (xi, yi, ti, co), f'{masks.shape=} != ({xi=}, {yi=}, {ti=}, {co=})'
    assert len(class_names) == co, f'{len(class_names)=} != {co=}'
    assert len(feature_names) == fi, f'{len(feature_names)=} != {fi=}'
    assert backend in ['numpy', 'matplotlib'], f'{backend=} not in ["numpy", "matplotlib"]'
    if xlabels is not None: assert len(xlabels) == ti, f'{len(xlabels)=} ({xlabels=}) != {ti=}'
    
    #RGB channel assertions
    assert in_channels_rgb==[] or all([ci in feature_names for ci in in_channels_rgb]) and len(in_channels_rgb)==3,\
        f'Not all {in_channels_rgb=} were found in input feature names {feature_names} or {len(in_channels_rgb)=}!=3'
    assert out_channels_rgb==[] or all([co in class_names for co in out_channels_rgb]) and len(out_channels_rgb)==3,\
        f'Not all {out_channels_rgb=} were found in outuput class names {class_names} or {len(out_channels_rgb)=}!=3'
    
    #Compute idx of input channels to plot
    idx_in_plot= [i for i,ci in enumerate(feature_names) if ci not in in_channels_rgb]
    #Compute idx of output channels to plot
    idx_out_plot= [i for i,co in enumerate(class_names) if co not in out_channels_rgb]
    #Compute final row names
    rgb= lambda names, rgb_names: ([n for n in names if n not in rgb_names] + ['RGB']) if rgb_names!=[] else names
    row_names= ( [f'{m} {co}' for m in ['True', 'Pred.'] for co in rgb(class_names, out_channels_rgb)] +\
         reduce_lists([[f'Input {fi}'] + [f'Pred. {fo} attrib. wrt input {fi}' 
                    for fo in rgb(class_names, out_channels_rgb)] 
                        for fi in rgb(feature_names, in_channels_rgb)]) )

    #We will plot the rows in the following order: y, yp, x(ci=0), a(ci=0, co=0), a(ci=0, co=1), ..., x(ci=1), ....
    #x y t c -> t c x y for labels, predictions, inputs;  co x y t ci -> t co ci x y for attributions
    #We also add the channel dimension
    a_plot= attributions
    l_plot_t, p_plot_t= labels.transpose(2,3,0,1)[:,None], predictions.transpose(2,3,0,1)[:,None], 
    t_plot_t, a_plot_t= inputs.transpose(2,3,0,1)[:,None], a_plot.transpose(3,0,4,1,2)[:,None]
    m_plot_t= masks.transpose(2,3,0,1)[:,[0]] #Keep only one of masks' channel dimensions

    #Transform data to rgb if needed. Attributions are computed as the mean over rgb channels
    if in_channels_rgb!=[]:
        rgb_idx_in= [feature_names.index(fi) for fi in in_channels_rgb]
        t_plot_rgb= [t_plot_t[:,[0],rgb_idx_in]] #t 1 c x y
        a_plot_t= np.concatenate([a_plot_t, #[:,:,:,idx_in_plot],  #t 1 co _ci_ x y for a_plot
                                  a_plot_t[:,:,:,rgb_idx_in].mean(axis=3, keepdims=True)], axis=3) 
    else: t_plot_rgb= []

    if out_channels_rgb!=[]:
        rgb_idx_out= [class_names.index(fo) for fo in out_channels_rgb]
        l_plot_rgb, p_plot_rgb= [l_plot_t[:,[0],rgb_idx_out]], [p_plot_t[:,[0],rgb_idx_out]] #t 1 c x y
        a_plot_t= np.concatenate([a_plot_t, #[:,:,idx_out_plot],  #t 1 co _ci_ x y for a_plot
                                  a_plot_t[:,:,rgb_idx_out].mean(axis=2, keepdims=True)], axis=2) 
    else: l_plot_rgb, p_plot_rgb= [],[]

    #Create list of images to plot 
    rgb_i= lambda idx, rgb_names: (idx + [idx[-1]+1]) if rgb_names!=[] else idx
    images= reduce_lists([
        [l_plot_t[:, :, idx] for idx in idx_out_plot] + l_plot_rgb, 
        [p_plot_t[:, :, idx] for idx in idx_out_plot] + p_plot_rgb,
        reduce_lists([ [t_plot_t[:, :, idx_i] if idx_i < t_plot_t.shape[2] else t_plot_rgb[0]] + 
                       [a_plot_t[:, :, idx_o, idx_i] for idx_o in rgb_i(idx_out_plot, out_channels_rgb)] 
                          for idx_i in rgb_i(idx_in_plot, in_channels_rgb) ])
        ])

    #Plot them to an rgb array
    if PAPER_STYLE: backend='matplotlib'
    arr= plot_maps(
        images= images,
        masks=[m_plot_t]*len(images), #Repat mask len(images) times
        cmaps=list(map(_get_plot_cmap, row_names)), 
        centers=list(map(_get_plot_center, row_names)),
        limits= list(map(_get_plot_limits, row_names)),
        ylabels= row_names, 
        xlabels= xlabels if xlabels is not None else list(range(len(t_plot_t))),
        mask_kwargs=dict(colors= {0:None, 1:'r'}), figsize=(19,10), backend=backend,
        numpy_backend_kwargs={'size':13, 'color':'black', 
                              'xstep':4, 'ystep':1, 
                              'labels':'grid', 'font':'OpenSans_Condensed-Regular.ttf'},
        max_text_width=20,
        cmap_kwargs=dict(nan_color=None, inf_color=None, zero_color=None), #Do not color bad pixels
        classes=None if PAPER_STYLE else {1:'Masked outputs'},
        title=None if PAPER_STYLE else None,
        **kwargs)

    #Save figure
    if save_path is not None:
        path= Path(save_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if backend == 'matplotlib' or PAPER_STYLE:
            fig, axes= arr
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            cv2.imwrite(str(path), arr[...,[2,1,0]])
            
    return arr