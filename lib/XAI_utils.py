from typing import List, Tuple, Optional, Union, Dict
import numpy as np
import copy
import torch
from tqdm.auto import tqdm

def aggregate_inputs(attributions, inputs, x_shapes, y_shape, agg_inputs, agg_inputs_function='mean'):
    #Select the aggregation function
    def agg_fn(x, axis=None):
        if 'abs' in agg_inputs_function: 
            x= np.abs(x)
        if axis == []:
            return x
        if 'mean' in agg_inputs_function: 
            return np.nanmean(x, axis=axis)
        x_max= np.nanmax(x, axis=axis)
        if 'max' in agg_inputs_function and agg_inputs_function != 'minmax': 
            return x_max
        x_min= np.nanmin(x, axis=axis)
        if 'min' in agg_inputs_function and agg_inputs_function != 'minmax': 
            return x_min
        if agg_inputs_function == 'minmax':
            max_bigger_than_min= x_max > -x_min
            return x_max * max_bigger_than_min + x_min * (~max_bigger_than_min)
        raise AssertionError(f'Unknown {agg_inputs_function=}. Must be one of '\
                             '[max, min, minmax, mean, absmean, absmax]')

    #Some checks
    print(f' > Aggregating inputs according to {agg_inputs=} with {agg_inputs_function=}'\
          f'\n - Initial attribution shapes: {[next(iter(a.values())).shape for a in attributions]} '\
          f'\n - Initial input shapes: {[next(iter(a.values())).shape for a in inputs]}'\
          f'\n - {x_shapes=}')
    assert len(agg_inputs) == len(x_shapes)
    for i, (agg, xs) in enumerate(zip(agg_inputs, x_shapes)): 
        assert not len(agg) or max(agg) < len(xs) - 1, \
            f'For input {i} with shape {xs}, you cannot aggregate indices {agg} '\
            'because you are either aggregating the feature dimension, or a dimension that does not exist'

    #Perform aggregation
    #Things that need no changes: labels, predictions, masks, y_shape, class_names
    attributions= [{k: agg_fn(v, axis=tuple(agg_i + len(y_shape) for agg_i in agg))
                   for i, (k,v) in tqdm(enumerate(a.items()), leave=False, 
                        desc=f'Processing attributions. Input {i+1}/{len(attributions)}',
                        total=len(attributions[0]))} 
                   for i0, (a, agg) in enumerate(zip(attributions, agg_inputs))]
    inputs= [{k: agg_fn(v, axis=tuple(agg))
             for i, (k,v) in tqdm(enumerate(a.items()), leave=False,
                  desc=f'Processing inputs. Input {i0+1}/{len(inputs)}', total=len(inputs[0]))} 
             for i0, (a, agg) in enumerate(zip(inputs, agg_inputs))]
    x_shapes= [[xi for i, xi in enumerate(x) if i not in agg] for x, agg in zip(x_shapes, agg_inputs)]
        
    print(f' - Final attribution shapes: {[next(iter(a.values())).shape for a in attributions]}'\
          f'\n - Final input shapes: {[next(iter(a.values())).shape for a in inputs]}'\
          f'\n - {x_shapes=}')
    
    return attributions, inputs, x_shapes

def ignore_classes(attributions, labels, predictions, masks, y_shape, class_names, 
                   agg_classes, ignore_classes_base, full=False):
    #First, if using -full, we need to expand the indexes to ignore, and define the list of indexes to keep!
    true_class_count= next(iter(predictions.values())).shape[-1]
    expanded_class_count= y_shape[-1]
    keep_classes= [cl for cl in range(true_class_count) if cl not in ignore_classes_base]
    if full:
        repeats= expanded_class_count // true_class_count
        assert repeats == len(agg_classes)
        keep_classes_full= [c for c in range(expanded_class_count) if c not in ignore_classes_extended]
    else:
        keep_classes_full= keep_classes
    class_names= list(np.array(class_names)[keep_classes_full])

    #Keep all those classes!
    #Things that need no changes: inputs, x_shapes
    print(f' > Keeping {class_names=} ({keep_classes=}) given {expanded_class_count=}'
          f', {true_class_count=}, and {ignore_classes_base=}')
    if len(y_shape) == 1:
        attributions= [{k:v[keep_classes_full] for k,v in tqdm(a.items(), leave=False, 
            desc=f'Processing attributions. Input {i+1}/{len(attributions)}')} for i,a in enumerate(attributions)]
    elif len(y_shape) == 2:
        attributions= [{k:v[:,keep_classes_full] for k,v in tqdm(a.items(), leave=False, 
            desc=f'Processing attributions. Input {i+1}/{len(attributions)}')} for i,a in enumerate(attributions)]
    else: 
        raise NotImplementedError('This should be generally implemented, not just checking every case')
        
    #Apply
    labels= {k:np.repeat(v[...,keep_classes], len(keep_classes_full), -1) for k,v in 
             tqdm(labels.items(), desc=f'Processing labels', leave=False)}
    predictions= {k:np.repeat(v[...,keep_classes], len(keep_classes_full), -1) for k,v in 
             tqdm(predictions.items(), desc=f'Processing predictions', leave=False)}
    masks= {k:np.repeat(v[...,[0]], len(keep_classes_full), -1) for k,v in 
             tqdm(masks.items(), desc=f'Processing masks', leave=False)}
    y_shape= y_shape[:-2] + [len(keep_classes_full)]
    
    return attributions, labels, predictions, masks, y_shape, class_names

def get_agg_mask(agg_mode:str, y:torch.Tensor, y_labels:torch.Tensor, sample:Dict[str, torch.Tensor], 
                 events:Optional[torch.Tensor]=None, error_threshold:float=0.5) -> torch.Tensor:
    '''
    Compute the aggregation mask according to the `agg_mode`
    
    :param agg_mode: It selects the aggregation mode. 
        It must be one of ["none", "events[-full]", "correctness[-full]", "labels[-full]", "custom[-full]"]
    :param y: Model predictions
    :param y_labels: Labels
    :param sample: The dictionary returned by the dataloader containing the batch
    :param events: tensor with the same shape as model outputs `y` indicating to what event that output corresponds
        (or 0 if it corresponds to no event). It canot be None if agg_mode == "events[-full]"
    :param error_threshold: Threshold to be used for agg_mode == 'correctness[-full]'
    
    :return: torch tensor of the final mask to be used for masked aggregation
    '''
    if agg_mode == 'none':
        mask=None
    elif agg_mode == 'events' or agg_mode == 'events-full':
        #The mask selects the output predictions given the threshold for each class
        mask= torch.Tensor(events > 0).to(y.device)
        mask= torch.concatenate([torch.logical_not(torch.sum(mask,dim=1,keepdim=True)>0), mask], axis=1).bool()
        if agg_mode == 'events-full': 
            mask= torch.any(mask, dim=1, keepdim=True)
            mask= torch.concatenate([mask, ~mask], axis=1) 
    elif agg_mode == 'correctness' or agg_mode == 'correctness-full':
        #The mask selects the output predictions whose abosulute error is below a threshold
        mask= torch.abs(y-y_labels) <= error_threshold
        if agg_mode == 'correctness-full': 
            mask= torch.any(mask, dim=1, keepdim=True)
            mask= torch.concatenate([mask, ~mask], axis=1) 
    elif agg_mode == 'labels' or agg_mode == 'labels-full':
        mask= y_labels.clone().type(torch.bool)
    elif agg_mode == 'custom' or agg_mode == 'custom-full':
        #Checks
        assert 'agg_mask' in sample.keys(), \
            f'If using {agg_mode=}, the aggregation mask must be accesible as sample["agg_mask"] for '\
            f'each sample produced by the dataloader'
        if agg_mode == 'custom-full':
            pass
            #assert sample['agg_mask'].shape == y.shape, f'{sample["agg_mask"].shape=} must be equal to {y.shape=}'
        else:
            assert all([si == sj for i, (si, sj) in enumerate(zip(sample['agg_mask'].shape, y.shape)) if i!=1]),\
                f'{sample["agg_mask"].shape} must be equal to {y.shape=} for all axis but axis 1 (out features)'
        #Get the mask
        mask= sample["agg_mask"]
    else:
        raise AssertionError(f'{agg_mode=} must be one of ["none", "events[-full]", '\
                             '"correctness[-full]", "labels[-full]", "custom[-full]"]')
        
    return mask

def event_at_positon(arr:np.ndarray, t:int, position:str='end') -> np.ndarray: 
    '''
        Takes an `arr` of shape (*), and creates a new one of shape (t, *)
                
        This is used for processing arrays before passing them to `plot_attributions_1d`, in the 
        case where there is a sinle output (instead of one output for every time step)
        
        :param arr: array to transform, shape (*)
        :param t: dimensionality of the first dimension of arr after processing
        :param position: where to place the original array with respect to the final array
            position == 'end': the original `arr` is at the last position of the 0th dimension,
                and the rest of the elements are zero.
            position == 'beginning': the original `arr` is at the first position of the 0th dimension,
                and the rest of the elements are zero
            position == 'all': the original `arr` is repeated t times over t index 0
        
        :return: transformed array with shape (t, *)
    '''
    if position == 'end':
        return np.concatenate([np.zeros([t-1, *arr.shape]), arr[None]], axis=0)
    elif position == 'beginning':
        return np.concatenate([arr[None], np.zeros([t-1, *arr.shape])], axis=0)
    elif position == 'all':
        return np.concatenate([arr[None]]*t, axis=0)
    else:
        raise AssertionError(f'{position=} must be one of ["end", "beginning", "all"]')
        
def clip_outliers(attributions:Union[np.ndarray, Tuple[np.ndarray]], outlier_perc:float=1, max_sample:int=5e6):
    '''
        Standarzation method that normalizes an array between 0 and 1, clipping out outliers
        
        :param attributions: array or tuple of arrays to standarize
        :returns: standarized array, a_min, a_max
    '''
    #Checks
    if outlier_perc is None: return attributions
    assert 0. <= outlier_perc <= 100., f'{outlier_perc=} not in [0., 100.]'

    #Transform to tuple
    is_tuple= isinstance(attributions, tuple)
    if not is_tuple: attributions= (attributions,)
    
    #Compute percentiles
    attr_flat= np.concatenate([a.flatten() for a in attributions])
    if len(attr_flat) > max_sample:
        attr_flat= np.random.choice(attr_flat, size=int(max_sample), replace=False)
    attr_min, attr_max= np.percentile(attr_flat, [outlier_perc/2, 100-outlier_perc/2])
    
    #Apply normalization to tuple elements
    attributions_final= []
    for a in attributions:
        attr= np.copy(a)
        attr[attr < attr_min]= attr_min
        attr[attr > attr_max]= attr_max
        attr/= (attr_max - attr_min)
        attributions_final.append(attr)
        
    #Return
    return tuple(attributions_final) if is_tuple else attributions_final[0], attr_min, attr_max

def create_baselines(x, baselines_config):
    """
    Create baselines for XAI methods based on configuration.
    
    :param x: Input tensor or tuple of tensors
    :param baselines_config: Configuration for baselines, can be:
                            - None: use zeros
                            - List of indices or None for each input tensor
    :return: Baseline tensor or tuple of baseline tensors
    """
    if isinstance(x, tuple):
        result = []
        for i, xi in enumerate(x):
            if i < len(baselines_config):
                if baselines_config[i] is None:
                    result.append(torch.zeros_like(xi))
                else:
                    # Use specified channels as baseline
                    indices = baselines_config[i]
                    baseline = torch.zeros_like(xi)
                    for j, idx in enumerate(indices):
                        if idx is not None:
                            baseline[:, j] = xi[:, idx]
                    result.append(baseline)
            else:
                result.append(torch.zeros_like(xi))
        return tuple(result)
    else:
        return torch.zeros_like(x)