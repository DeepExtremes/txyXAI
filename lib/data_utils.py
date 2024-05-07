import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import seaborn as sns
from scipy.interpolate import make_interp_spline
from pathlib import Path
import matplotlib as mpl

#Custom imports
from .visualization import plot_maps

#Global configuration
NDVI='kNDVI' #['NDVI' or 'kNDVI']
FONT_SIZE= 20

#Get the nth element (first by default) of every tuple in a list of tuples
get_nth= lambda l, n=0: [i[n] for i in l]

def setup_matplotlib():
    'Reset matplotlib config, change defaults, and get default list of colors. Create RdGn cmap'
    #Reset matplotlib config, change defaults
    mpl.rcParams.update(mpl.rcParamsDefault)    
    plt.style.use('ggplot')
    mpl.rcParams.update({'font.size': 12})
    
    #Create colormap for attribution plotting and register it
    #https://stackoverflow.com/questions/38246559/how-to-create-a-heat-map-in-python-that-ranges-from-green-to-red
    try:
        c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
        v = np.array([0,.15,.4,.5,0.6,.9,1.])
        cmap= mpl.colors.LinearSegmentedColormap.from_list('RdGn', list(zip(v,c)), N=256)
        plt.register_cmap('RdGn', cmap)
    except Exception as e: 
        #print(e)
        pass
    
    #We apparently run out of colors, let's create some more just in case
    base_color_list= list(plt.rcParams['axes.prop_cycle'].by_key()['color']) #list(mpl.colors.TABLEAU_COLORS)
    color_list= base_color_list + [sns.desaturate(c, 0.2) for c in base_color_list] + \
                                  [sns.desaturate(c, 0.4) for c in base_color_list]
    
    return color_list

#Processing methods
def interpolate_clima(all_clima, all_clima_count, s, dom, bins, interp_order=1):
    #Check the number of bins
    assert bins == all_clima.shape[1]
    assert s >= 0 and s < bins, f'{s=} must be in [0, {bins-1}]'

    #Compute s and dom for pred timestep (s= month [0-11], dom= day of month [0-1])
    s_before, s_after= s-1 if s > 0 else bins-1, s+1 if s < bins-1 else 0
    dom= 2*dom -1 #dom1 € [-1,1]: 0->middle of month, -1->begin, 1->end

    #Get clima for s_before and s_after
    clima= all_clima[:,s] / (all_clima_count[:,s] + 1e-6)
    valid= all_clima_count[0, s] > 0
    clima_before= all_clima[:,s_before] / (all_clima_count[:,s_before] + 1e-6)
    valid_before= all_clima_count[0,s_before] > 0
    clima_after= all_clima[:,s_after] / (all_clima_count[:,s_after] + 1e-6)
    valid_after= all_clima_count[0,s_after] > 0

    #Get output array
    all_curr_clima_out= np.zeros_like(clima)

    #Interpolate either linearly or quadratically
    if int(interp_order) == 1:
        if dom < 0: 
            dom*= -1 
            clima_interp= clima_before*dom/2 + clima*(1-dom/2)
        else:
            clima_interp= clima_after*dom/2 + clima*(1-dom/2)
    elif int(interp_order) == 2:
        clima_3m= np.stack([clima_before, clima, clima_after], axis=0)
        interp_fn= make_interp_spline([-2,0,2], clima_3m, k=2)
        clima_interp= interp_fn([dom])[0]
    else:
        raise ValueError(f'{interp_order} can only be in [1,2]. Further degrees are not supported yet')

    #Add results and try to fix missing values
    if dom < 0: 
        all_curr_clima_out+= clima_interp*(valid&valid_before)
        all_curr_clima_out+= clima_before*(~valid&valid_before)
        all_curr_clima_out+= clima*(valid&~valid_before)
    else:
        all_curr_clima_out+= clima_interp*(valid&valid_after)
        all_curr_clima_out+= clima_after*(~valid&valid_after)
        all_curr_clima_out+= clima*(valid&~valid_after)

    return all_curr_clima_out

def ndvi(red, nir, eps=1e-4, limits=(-1.,1.), kndvi=NDVI=='kNDVI'):
    ndvi= (nir - red) / (nir + red + eps)
    ndvi[ndvi<limits[0]]=limits[0]
    ndvi[ndvi>limits[1]]=limits[1]
    if kndvi: 
        if isinstance(ndvi, np.ndarray): return np.tanh(ndvi**2)
        else: return torch.tanh(ndvi**2)
    else: return ndvi

def _process_mask(arr, class_indices, processing_info):
    ''' Takes an integer mask array (several concatenated along dimension 0 of len C), a list of 
        all classes in each mask (len C), and a list of processing_info (len C), and preprocesses
        everything according to the following criteria. For each mask array at position i, assuming it 
        has classes 0,1,2,3,4:
            if processing_info[i] = [1,3]: Set to one if class is 1 or 3
            if processing_info[i] = [None]: Convert all classes to one hot and remove first dimension, 
                similar to [[1,2,3,4]] (notice double list)
            if processing_info[i] = [[1,2], [3,4]]: Get a one hot array with first entry for classes 1 or 2,
                and second entry for classes 3 or 4 
    '''
    arr_list= []
    for i, (ci, pinfo) in enumerate(zip(class_indices, processing_info)):
        if pinfo is None:
            arr_list.append(np.stack([arr[i] == c for c in ci[1:]], axis=0))
        elif isinstance(pinfo[0], list):
            arr_list.append(np.stack([np.isin(arr[i], pinfo_i) for pinfo_i in pinfo], axis=0))
        else:
            arr_list.append(np.sum([arr[i] == c for c in pinfo], axis=0) > 0)
    return np.concatenate(arr_list, axis=0)

def _standarize(arr, minmax, K=1):
    'Standarizes by first clipping to minmax, and then dividing by max-min'
    assert len(arr) == len(minmax), f'{len(arr)=} != {len(minmax)=}'
    for i, minmax in enumerate(minmax):
        if isinstance(minmax, tuple): 
            vmin, vmax= minmax
            # arr[i, arr[i]<vmin]= vmin
            # arr[i, arr[i]>vmax]= vmax
            # arr[i]/= (vmax - vmin)
            vrange= vmax - vmin
            vmin_cutoff, vmax_cutoff= vmin - K*vrange, vmax + K*vrange
            arr[i, arr[i] < vmin_cutoff]= vmin_cutoff
            arr[i, arr[i] > vmax_cutoff]= vmax_cutoff
            arr[i]= (arr[i] - vmin)/vrange
        elif isinstance(minmax, float):
            arr[i]/= minmax
        elif isinstance(minmax, str) and minmax == 'std':
            old_arr= arr[i].copy() #Store old array
            #Compute std causally: DO NOT USE, it is unstable, specially at the beginning
            for t in range(len(old_arr)):
                arr[i,t]/= (np.std(old_arr[:t+1]) + 1e-6)
        else:
            raise AssertionError(f'{minmax=} must be a tuple of two floats, a float')
    return arr

def _auto_scale(variables, minicube, dims=(128, 128), **upscale_kwargs):
    'Read all variables in `variables` from `minicube`, upscaling them to (lon, lat) if necessary'
    if len(variables):
        variables_data= []
        for v in variables:
            v_data= minicube[v].values
            if v_data.shape[-2] != dims[0] or v_data.shape[-1] != dims[1]:
                v_data= _scale(v_data, dims, **upscale_kwargs)
            variables_data.append(v_data)
        return np.stack(variables_data, axis=0)
    else:
        return None #Could be omitted

def _scale(x, dims=(128, 128), method='nn'):
    'Scale x to dimensions dims using `method`'
    assert method in ['nn', 'linear', 'cubic']
    cv2_method= {'nn':cv2.INTER_NEAREST, 'linear':cv2.INTER_LINEAR, 'cubic':cv2.INTER_CUBIC}[method]
    #If x is an image-like array, process it directly
    if len(x.shape) == 2 or len(x.shape) == 3 and x.shape[0] in [1,3,4]:
        return cv2.resize(x, dims, 0, 0, interpolation=cv2_method)
    else: #Otherwise, split it and handle each image-like array separately
        x_res= np.reshape(x,[-1, x.shape[-2], x.shape[-1]])
        x_res= np.stack([cv2.resize(xi, dims, 0, 0, interpolation=cv2_method) for xi in x_res], axis=0)
        return np.reshape(x_res,[*x.shape[:-2], dims[-2], dims[-1]])

#Plotting methods
def _get_plot_cmap(var_name):
    'Returns a cmap for plotting dependig on the name of the variable'
    if 'attr' in var_name.lower(): return 'RdGn'
    #elif 'rgb' in var_name.lower(): return 'hot'
    elif 'ndvi' in var_name.lower() or 'dev' in var_name.lower(): return 'RdYlGn'
    else: return 'hot'

def _get_plot_center(var_name):
    'Returns a `center` mode plotting dependig on the name of the variable'
    return 'ndvi' in var_name.lower() or 'mean' in var_name.lower() or 'attr' in var_name.lower()

def _get_plot_limits(var_name):
    'Returns the plotting limits of a variable'
    if 'rgb' in var_name.lower() and 'ndvi' not in var_name.lower() and \
       'attr' not in var_name.lower() and 'mean' not in var_name.lower() and\
       'detrend' not in var_name.lower(): return (0, 0.3)
    elif 'rgb' in var_name.lower() and 'detrend' in var_name.lower(): return (-0.3,0.3)
    #elif 'ndvi' in var_name.lower(): return (-0.7, 0.7)
    #elif 'dev' in var_name.lower(): return (-1, 1)
    #elif 'attr' in var_name.lower(): return (-1, 1)
    elif 'input' in var_name.lower(): return (0, 1)
    else: return (-1, 1.)

def _npdate2str(date):
    return np.datetime_as_string(date, unit="D")

def _str2npdate(date):
    return np.array(date, dtype='datetime64[D]')

def plot_clima(all_clima, all_clima_count, t_unique, compute_ndvi):
        clima_plot= np.swapaxes(all_clima / (all_clima_count + 1e-6), 0,1)
        fig, axes= plot_maps(
                    images=[clima_plot[:,[2,1,0]], clima_plot[:,[3]]] +\
                           ([clima_plot[:,[4]]] if compute_ndvi else []) + [all_clima_count[0,:,None]], 
                    xlabels=t_unique+1, 
                    ylabels=([f'{v}_clima' for v in ['RGB', 'B8A'] +\
                            ([f'{NDVI}'] if compute_ndvi else [])]) + ['# valid samples'], 
                    figsize=(32,7), 
                    limits=[(0,0.3), (0,1.)] +\
                           ([(0,1.)] if compute_ndvi else []) + [(0, all_clima_count.max())], 
                    title=f'All Climatology', 
                    cmaps=[None, 'hot'] +\
                           (['RdGn'] if compute_ndvi else []) + ['gist_gray'], 
                    matplotlib_backend_kwargs={'text_size':FONT_SIZE})
        return fig, axes

def plot_txy(txy_real, txy_mask, t, plot_idx, plot_names, title='', select=slice(50,150,7),
             labels_t=None):
    'Plot txy_real and txy_mask data'
    #Notice: c t lon lat -> t c lon lat
    backend= 'numpy'
    try: 
        if (select.stop - select.start)/select.step < 30: backend= 'matplotlib'
    except: 
        pass
    
    xlabels= list(map(lambda i_d: f'{i_d[0]}\n{_npdate2str(i_d[1])}', enumerate(t)))
    #if labels_t is not None: xlabels= [f'{x}\n[{l}]' for x,l in zip(xlabels, labels_t)]
    stuff= plot_maps(
        images=[txy_real.transpose(1,0,2,3)[select, idx] for idx in plot_idx], 
        masks=[txy_mask.transpose(1,0,2,3)[select]]*len(plot_idx), #Repat mask len(images) times
        cmaps=list(map(_get_plot_cmap, plot_names)), 
        centers=list(map(_get_plot_center, plot_names)),
        limits= list(map(_get_plot_limits, plot_names)),
        ylabels=plot_names, 
        xlabels=xlabels[select],
        mask_kwargs=dict(colors= {0:None, 1:'r'}), classes= {1:'Invalid'},
        title=title, backend=backend,
        numpy_backend_kwargs={'size':13, 'color':'black', 'xstep':4,
                              'labels':'grid', 'font':'OpenSans_Condensed-Regular.ttf'},
        plot_mask_channel=0, matplotlib_backend_kwargs={'text_size':FONT_SIZE},
        figsize=(27.5,10),
                    )
    return stuff

def plot_xy(rgb, xy_real, xy_cat, t, plot_idx, plot_names, classes, select=slice(None,None,25), title=''):
    'Plot tx_real and tx_cat data against rgb'
    #Notice: c t lon lat -> t c lon lat
    rgb_proccessed= rgb.transpose(1,0,2,3)[select]
    backend= 'numpy'
    try: 
        if (select.stop - select.start)/select.step < 30: backend= 'matplotlib'
    except: 
        pass
    stuff= plot_maps(
        images=[rgb_proccessed] + [xy_real[[i]][None].repeat(
            rgb_proccessed.shape[0], axis=0) for i in range(xy_real.shape[0])], 
        masks=[xy_cat.argmax(axis=0)[None].repeat(
            rgb_proccessed.shape[0], axis=0)[:,None]]*(1 + xy_real.shape[0]),
        cmaps=list(map(_get_plot_cmap, plot_names)), 
        centers=list(map(_get_plot_center, plot_names)),
        limits= [None] * len(plot_names),
        ylabels=plot_names, 
        xlabels=list(map(lambda i_d: f'{i_d[0]}\n{_npdate2str(i_d[1])}', enumerate(t)))[select],
        classes={i:v for i, (k,v) in enumerate(classes.items())}, 
        title=title, figsize=(27,6), backend=backend, 
        numpy_backend_kwargs={'size':13, 'color':'black', 'xstep':4,
                              'labels':'grid', 'font':'OpenSans_Condensed-Regular.ttf'},
        plot_mask_channel=0, matplotlib_backend_kwargs={'text_size':FONT_SIZE},
        colorbar_position='vertical'
        )
    return stuff

def plot_t(plot_names, t_real, t, labels_t=None, columns=4):
    'Plot t data'
    rows= len(plot_names) // columns
    rows= rows if len(plot_names) % columns == 0 else rows + 1
    fig, axes = plt.subplots(rows, columns, figsize=(columns*5, rows*3))
    for i, (name, var) in enumerate(zip(plot_names*2, t_real)):
        ax= axes[i//columns, i%columns] if len(axes.shape) == 2 else axes[i]
        ax.plot(t, var); ax.set_title(name)
        if labels_t is not None:
            unique_events= np.unique(labels_t)[1:]
            for event_n in unique_events:
                event_idx= labels_t==event_n
                ax.plot(t[event_idx], var[event_idx], '.-', label=str(event_n))
            if not i and len(unique_events): ax.legend()
    fig.tight_layout()
    return fig, axes

#Functions used by the model
def plot_prediction(x, masks, labels, y_pred, 
                    t_y=None, t_data=None, t_events=None, t_plot_names=None, 
                    select=np.s_[150:200:1], title='', N='all', save_path=None, add_index=True, ids=None,
                    naive=None, plot_infrared=True, backend='numpy'): 
    '''
        Plots the first `N` elements of the predicted batch (or all if N='all').
        It expects inputs `x`, `masks`, `labels` and `y_pred` to have dimensions (b c t lon lat)
        and `t_y` to have dimension (t)
    '''
    #Generate custom plots for the paper
    # select=np.s_[238:252:1]
    # backend= 'matplotlib'
    
    if N=='all': N=len(x)
    N= min(N, len(x))
    if ids is None: ids= ['']*N
    for i, cid in enumerate(ids):
        #Prepare data: b c t lon lat -> t c lon lat
        y_plot= np.transpose(labels[i], (1,0,2,3))
        yp_plot=  np.transpose(y_pred[i], (1,0,2,3))
        m_plot= np.transpose(masks[i], (1,0,2,3))
        t_y_plot= t_y if not add_index or t_y is None else [f'{i}\n{l}' for i, l in enumerate(t_y)]
        if yp_plot.shape[1] == 1:
            ndvi= y_plot
            ndvip= yp_plot
            plot_idx, plot_names= [[0]], [NDVI]
        else:
            if yp_plot.shape[1] == 4:
                y_plot= np.concatenate([y_plot, _ndvi(y_plot[:, [2]], y_plot[:, [3]])], axis=1)
                yp_plot= np.concatenate([yp_plot, _ndvi(yp_plot[:, [2]], yp_plot[:, [3]])], axis=1)
            else:
                pass
            if plot_infrared:
                plot_idx, plot_names= [[2,1,0], [3], [4]], ['RGB', 'B8A', NDVI]
            else:
                plot_idx, plot_names= [[2,1,0], [4]], ['RGB', NDVI]

        m_plot_final= np.max(m_plot[select], axis=1, keepdims=True).astype(int)
        if t_events is not None: 
            t_events_plot= t_events[i] #t
            t_data_plot= t_data[i]
            m_plot_final[t_events_plot[select] > 0]= 2 #Create a class for events
            t_y_plot= [f'{ty} [{e}]' if e!=0 else ty for ty, e in zip(t_y_plot, t_events_plot)]

        #Plot maps
        pred_plot_idx, pred_plot_names= plot_idx, [f'pred. {n}' for n in plot_names]
        id= cid.replace("/","--")
        title_plot= f'{title} {id=} {naive=} events={list(np.unique(t_events_plot)[1:][-15:])} {i=} {select=}'
        arr= plot_maps(
              images=[*[y_plot[select, idx] for idx in plot_idx],
                      *[yp_plot[select, idx] for idx in pred_plot_idx]],
              masks=[m_plot_final]*(len(plot_idx) + len(pred_plot_idx)),
              cmaps=list(map(_get_plot_cmap, plot_names + pred_plot_names)),
              centers=list(map(_get_plot_center, plot_names + pred_plot_names)),
              limits=list(map(_get_plot_limits, plot_names + pred_plot_names)),
              ylabels=plot_names + pred_plot_names, 
              xlabels=t_y_plot[select] if t_y_plot is not None else None,
              mask_kwargs=dict(colors= {0:None, 1:'r', 2:'b'}), classes= {1:'Bad', 2:'Event'}, #title=title_plot,
              backend=backend, numpy_backend_kwargs={'size':13, 'color':'black', 'xstep':1,
                                                     'labels':'grid', 'font':'OpenSans_Condensed-Regular.ttf'},
              figsize=(27.5,10),
              matplotlib_backend_kwargs={'text_size':FONT_SIZE-3},
              stack_every=73, #Stack every year (approx.) 73*5=365
            )

        if save_path is not None:                
            #Save
            path= Path(save_path).resolve()
            path.mkdir(parents=True, exist_ok=True)
            if backend == 'numpy':
                cv2.imwrite(str(path / f'{title_plot}.jpg'), arr[...,[2,1,0]])
            else:
                fig= arr[0]
                fig.savefig(path / f'{title}.jpg', dpi=120, bbox_inches='tight')
                plt.close(fig)

        #Plot weather
        if t_data is not None:
            t_y_plot_final= _str2npdate(t_y)
            fig, axes= plot_t(t_plot_names, t_data_plot, t_y_plot_final, 
                                           labels_t=t_events_plot, columns=4)
            if save_path is not None:
                path= Path(save_path).resolve()
                fig.savefig(path / f'{title_plot} - Weather.jpg', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
def prepare_batch(batch, num_classes, pred_period, device=None, has_batch_dimension=True):
    #Unpack batch
    #Actual batch data: (t, xy, txy) -> [8, 16, 495] [8, 36, 64, 64] [8, 10, 495, 64, 64]
    t, xy, txy= batch['t'], batch['xy'], batch['txy']

    #The dates of the timesamples converted to str (use the first of every tuple, because all in the batch are the same)
    t_dates_str= get_nth(batch['meta']['index']) if has_batch_dimension else batch['meta']['index']

    #Get the labels where events occur
    t_labels= batch['meta']['event_labels']

    #Index positions in txy where we have the labels of the output of the model
    txy_label_idx= get_nth(batch['meta']['txy_label_idx']) if has_batch_dimension else batch['meta']['txy_label_idx']

    #Index positions in xy where we have the info that we need for the NNSE metric (landcovers corresponding to vegetation)
    xy_label_metric_idx= get_nth(batch['meta']['xy_label_metric_idx'])\
                                 if has_batch_dimension else batch['meta']['xy_label_metric_idx']

    #txy mask (i.e. cloud mask). If True, sample is good!
    if has_batch_dimension:
        txy_mask= batch['txy_mask'].repeat(1, num_classes,1,1,1)==0 if 'txy_mask' in batch.keys()\
                                                                    else torch.ones(txy.shape) > 0
    else:
        txy_mask= batch['txy_mask'].repeat(num_classes,1,1,1)==0 if 'txy_mask' in batch.keys()\
                                                                 else torch.ones(txy.shape) > 0

    #Build inputs and outputs
    #Build _in, _out, and _mask_out
    if has_batch_dimension:
        t_in, xy_in, txy_in= t[:,:,:-pred_period], xy, txy[:,:,:-pred_period]
        txy_out, txy_mask_out= txy[:,txy_label_idx, pred_period:], txy_mask[:,:, pred_period:]
        t_labels= t_labels[None,:, pred_period:]
    else:
        t_in, xy_in, txy_in= t[:,:-pred_period], xy, txy[:,:-pred_period]
        txy_out, txy_mask_out= txy[txy_label_idx, pred_period:], txy_mask[:, pred_period:]
        t_labels= t_labels[pred_period:]

    #Build the mask that we need for the NNSE metric (landcovers corresponding to vegetation)
    if has_batch_dimension:
        xy_metric_masks_out= xy[:,xy_label_metric_idx].max(axis=1, keepdims=True).values > 0.5
    else:
        xy_metric_masks_out= xy[xy_label_metric_idx].max(axis=1, keepdims=True).values > 0.5

    #Rename to convention
    x, labels, masks, metric_masks= (t_in, xy_in, txy_in), txy_out, txy_mask_out, xy_metric_masks_out

    if device is not None:
        return tuple(xi.to(device) for xi in x), labels.to(device), masks.to(device), metric_masks.to(device), \
               t_labels, t_dates_str, t.to(device), xy.to(device), txy.to(device)
    else:
        return x, labels, masks, metric_masks, t_labels, t_dates_str, t, xy, txy