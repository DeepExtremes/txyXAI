from typing import List, Tuple, Optional, Union, Dict
import captum
from captum.attr import * # Saliency, InputXGradient, GuidedBackprop, IntegratedGradients, LayerGradCam, GuidedGradCam, LayerActivation, LayerAttribution, DeepLift, FeatureAblation
from itertools import product
import numpy as np
from tqdm.auto import tqdm
import copy
import torch 
from .XAI_utils import get_agg_mask, create_baselines
    
#Before, this was just x.T, but that is now deprecated
transpose_all= lambda x: x.permute(*torch.arange(x.ndim - 1, -1, -1))

def attribute(config:dict, model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, num_classes:int, 
              out_agg_dim:Optional[tuple], agg_mode:str, events:Optional[torch.Tensor]=None, debug:bool=False,
              ignore_classes:List[int]=[]) -> (Tuple[Dict[str,np.ndarray], 
                    Dict[str,np.ndarray], Dict[str,np.ndarray], Dict[str,np.ndarray], List[int], List[int]] ):
    '''
    Attribute over an arbitrary amount of input and output dimensions. It also supports aggregation 
    of the outputs (e.g. by mean) over specific dimensions, and aggregation of the outputs based on a mask 
    (e.g.: consider only predictions that have low error, consider only predicitions given a certain ground
    truth class, or based on a custom compartmentalization).
    
    :param config: Global configuration dictionary for the case, containig all hyperparameters. Here we use:
        config['xai']['out_agg_dim']: None or tuple, if tuple, output dims wrt which we aggregate the output
        config['xai']['type']: Captum attribution method (str)
        config['data']['data_dim'] & config['arch']['input_model_dim']: Used by `adapt_variables`
    :param model: Pytorch model
    :param dataloader: The dataloader used by the Pytorch model
    :param num_classes: number of output classes / features for the problem
    :param out_agg_dim: tuple of dimensions wrt wich we aggregate the output, or None, to perform no aggregation
    :param agg_mode: Used for masked aggregation, it selects the aggregation mode. 
            It must be one of ["none", "events[-full]", "correctness[-full]", "labels[-full]", "custom[-full]"]
    :param events: tensor with the same shape as model outputs `y` indicating to what event that output corresponds
        (or 0 if it corresponds to no event). It canot be None if agg_mode == "events[-full]"
    :param ignore_classes: List of output classes to ignore for attribution
    :param debug: if >= 1, it runs the XAI only for `debug` samples and prints some extra info
    
    :return: tuple containing:
        Dictionaries with a key for every attributed event with numpy arrays:
            attributions: array with shape ([out x, out y, out t], out classes, [in x, in y, in t], in features)
            inputs: array with shape ([in x, in y, in t], in features)
            labels: array with shape ([out x, out y, out t], [out classes])
            predictions: array with shape ([out x, out y, out t], out classes)
        List with the shape of x and y after processing:
            x_shape: List [[in x, in y, in t], in features]
            y_shape: List [[out x, out y, out t], out classes]
    '''
    # GPs
    try:    final_activation= config['implementation']['loss']['activation']['type']
    except: final_activation= 'none'
    
    #Using cross aggregation?
    cross_agg= agg_mode.endswith('-full')
    keep_classes= [c for c in range(num_classes) if c not in ignore_classes]
    was_nan= False
    DEBUGGED= False
        
    #If mask is provided, out_agg_dim is ignored, and the mask is used for aggregation instead
    if agg_mode != 'none' and out_agg_dim is not None:
        print(f'Warning: if {agg_mode=} != "none", then {out_agg_dim=} will be ignored')
        
    #Get model wrapper and xai method
    model_wrapper= XAIModelWrapper(model, out_agg_dim=out_agg_dim, num_classes=num_classes, 
                   cross_agg=cross_agg, using_mask=agg_mode!='none', 
                   is_GP=final_activation in ['ExactGP', 'ApproximateGP'], ignore_classes=ignore_classes)
    xai_method= globals()[config['xai']['type']](model_wrapper)
    
    #Iterate over samples and output dimensions (after aggregation)
    model.eval()
    y_shape= None
    #with torch.no_grad():
    #Notice that inputs and attributions will now be a list of dicts
    inputs, attributions, predictions, labels, masks, x_shapes= None, None, {}, {}, {}, None
    for index, sample in enumerate(pbar := tqdm(dataloader)): #It over events            
        #Get batch data
        if sample == {}: #Batch data was lost
            print(f'Warning: Dataloader returned no data for {index=}. Skipping')
            continue
            
        #if not 'masks' in sample.keys(): sample['masks'] = torch.ones(sample['x'].shape)
        #x, _, y_labels = adapt_variables(config, sample['x'], sample['masks'], sample['labels'])
        x, y_labels= sample['x'], sample['labels'] 
        if not isinstance(x, tuple): x= (x,) 
        for xi in x: xi.requires_grad= True
        #Create the proper list of dicts , but only do it in the first iteration
        if inputs is None: inputs, attributions= [{} for _ in x], [{} for _ in x]
        event_name = sample.get('event_name', index*x[0].shape[0] + np.arange(x[0].shape[0]))
        pbar.set_description(f'Explaining Dataloader: [{event_name=}]')

        #Get batch data and prediction
        if len(y_labels.shape) == 1: y_labels = y_labels.unsqueeze(dim=1)
        y = model.likelihood(model(*x)).mean.unsqueeze(-1) \
            if final_activation in ['ExactGP', 'ApproximateGP'] else model(*x)


        #If labels seem to be int-encoded, try to one-hot encode them
        #We need to move the output channel dimension at the end for this to work
        #A new dimension will be created at the end containing the one-hot encoded labels
        #Then we move it back to position 1
        if not y.shape == y_labels.shape:
            if index == 0: f' > {y.shape=} != {y_labels.shape}. Attempting to one-hot encode y_labels'
            y_labels= torch.eye(y.shape[1])[y_labels.swapaxes(1,-1)].swapaxes(1,-1)
            assert y.shape == y_labels.shape, \
               f'Attempted to one hot encode {y_labels.shape=} to make it like {y.shape=}, but something failed'

        #Create aggregation mask, which is just like y, but with (possibly) a different number of 
        #output channels (the agg. channels)
        mask= get_agg_mask(agg_mode, y, y_labels, sample, events=events[[index]] 
                           if agg_mode=='events' or agg_mode == 'events-full' else None)
        
        #Try to move everything to the correct device
        try:
            x= tuple(xi.to(model.device) for xi in x)
            if mask is not None: mask= mask.type(x[0].type()).to(model.device)
        except Exception as exception: 
            if index == 0: f' > Attempted to move data to model\'s device, but got {exception=}'

        #y_shape after aggregation (if used)
        if y_shape is None: 
            assert y.shape[1] in [num_classes, 1], f'Output classes must be located in axis 1 of y' 
            y_shape= list(model_wrapper.agg_y(y, mask=mask).shape[1:][::-1])

        #Create output arrays
        if debug and not DEBUGGED:
            print('XAI shapes before aggregation:')
            print(f' > {[xi.shape for xi in x]=}\n > {y.shape=}\n > {y_labels.shape=}')
            if mask is not None: print(f' > {mask.shape=}')

        x_shapes=[]
        for input_idx, xi in enumerate(x): #Iterate over input tuple
            x_shape= transpose_all(xi[0]).shape
            x_shapes.append(list(x_shape))
            for e in event_name:
                attributions[input_idx][e]= np.zeros([*y_shape,*x_shape]) 
                inputs[input_idx][e]= transpose_all(xi[0]).detach().cpu().numpy().astype(np.float32)
                y= torch.concatenate([1-y, y], axis=1) if y.shape[1]==1 and num_classes==2 else y
                predictions[e]= transpose_all(torch.squeeze(y, dim=0)).detach().cpu().numpy().astype(np.float32)\
                    if mask!=None else transpose_all(model_wrapper.agg_y(y, mask)[0]
                                                     ).detach().cpu().numpy().astype(np.float32)
                labels[e]= transpose_all(torch.squeeze(y_labels, dim=0)).detach().cpu().numpy().astype(np.float32)\
                    if mask!=None else transpose_all(model_wrapper.agg_y(y_labels, mask)[0]
                                                     ).detach().cpu().numpy().astype(np.float32)
                masks[e]= transpose_all(torch.squeeze(mask, dim=0)).cpu().numpy().astype(int) if mask!=None else None
                
                #Keep only some output classes
                if len(ignore_classes):
                    predictions[e]= predictions[e][...,keep_classes]
                    labels[e]= labels[e][..., keep_classes]
                    if mask is not None and not cross_agg and masks[e].shape[-1] != labels[e].shape[-1]: 
                        masks[e]= masks[e][..., keep_classes]
                
                #And repeat them as many times as custom classes in the cross aggregation
                if cross_agg:
                    predictions[e]= np.repeat(predictions[e], y_shape[-1], -1)
                    labels[e]= np.repeat(labels[e], y_shape[-1], -1)
                    if mask is not None and masks[e].shape[-1] != labels[e].shape[-1]:  
                        masks[e]= np.repeat(masks[e], y_shape[-1], -1)

        if debug and not DEBUGGED:
            print('XAI shapes after aggregation:')
            print(f' > {[xs for xs in x_shapes]=} ([in x, in y, in t], in features)')
            print(f' > {y_shape=} ([out x, out y, out t], out classes)')
            print(f' > {[a[event_name[0]].shape for a in attributions]=}'
                  ' ([out x, out y, out t], out classes, [in x, in y, in t], in features)')
            print(f' > {[i[event_name[0]].shape for i in inputs]=} ([in x, in y, in t], in features)')
            print(f' > {labels[event_name[0]].shape=} ([out x, out y, out t], [out classes])')
            print(f' > {predictions[event_name[0]].shape=} ([out x, out y, out t], out classes)')
            if mask is not None: print(f' > {masks[event_name[0]].shape=} ([out x, out y, out t], out classes)')
            DEBUGGED= False
                        
        #We atribute over all output dimensions after aggregation (if needed)
        default_attr= tuple(torch.clone(xi).detach()*np.nan for xi in x) #Set default attribution to nan
        for attr_target in product(*[range(y_dim) for y_dim in y_shape]):
            #Perform attribution mask is None, otherwise perform attribution only if mask has non-zero values 
            #for the specific output that is being attributed
            assert len(attr_target) == 1, f'The mask indexing below must be fixed for {len(attr_target)}>1'
            if mask is None or torch.any(mask[0, attr_target[0]]):
                #Let's create here the baselines for the XAI methods that need them
                baselines = None
                if config['xai']['type'] in ['GradientShap', 'IntegratedGradients']:
                    # Check if custom baselines are defined
                    if 'baselines' in config['xai'] and config['xai']['baselines'] is not None:
                        baselines = create_baselines(x, config['xai']['baselines'])
                    else:
                        baselines = torch.zeros_like(x) if len(x)==1 else tuple([torch.zeros_like(xi) for xi in x])
                if debug: 
                    print(f" > Running XAI method: {config['xai']['type']}, "
                        f"with parameters: {config['xai']['params']}")
                    
                attr= xai_method.attribute(x[0] if len(x)==1 else x, target=attr_target[::-1], 
                            additional_forward_args=mask, **config['xai']['params'], 
                            **({'baselines': baselines} if baselines is not None else {}))
            else: #Else set attribution to nan
                attr= tuple(torch.clone(a).detach() for a in default_attr)
                
            #Cannot directly index the array because that does not support the * operator
            for e_idx, e in enumerate(event_name):
                for input_idx, a in enumerate(attr if isinstance(attr, tuple) else [attr]):
                    attributions[input_idx][e].__setitem__(
                        attr_target, transpose_all(a[e_idx]).detach().cpu().numpy().astype(np.float32))
                    if a.isnan().any() and not was_nan: 
                        was_nan= True
                        print(f'Warning: NaN values found in attributions for {attr_target=} and {event_name=}'
                              '. Further warnings for the same problem will be silenced')

        if debug and index >= int(debug-1): break

    if x_shapes is None: 
        raise AssertionError('Attribution loop never ran: either your dataloader has no samples, or'
                             ' there is an Exception happening in your dataloader')
        
    return attributions, inputs, labels, predictions, masks, x_shapes, y_shape

class XAIModelWrapper(torch.nn.Module):
    def __init__(self, wrapped_model:torch.nn.Module, out_agg_dim:Optional[tuple]=(2,3), num_classes:int=-1, 
                 cross_agg:bool=False, agg_dim_fn=torch.mean, agg_mask_fn=torch.mean, 
                 using_mask:bool=False, is_GP:bool=False, ignore_classes:list=[]):
        '''
            This wrapper aggregates an output tensor over `out_agg_dim` using `agg_dim_fn`. 
            This makes it easier to attribute with respect to a single output scalar, as opposed 
            to individual pixel output attribution.

            Instead, if a mask is provided (not None), it uses this mask to aggregate over the classes,
            either by simple direct masking if `cross_agg=False` where the mask just selects some
            wanted pixels from each output class, or by generating all possible combinations of the output
            classes with the masking classes if `cross_agg=True`. In this last case, the new number 
            of classes is the product of original_classes x aggregation_classes. Selected samples are then
            aggreated using `agg_mask_fn`

            It also expands the output channel dimension if it has only a size of 1 and `num_classes`>1.
            E.g.: it expands from binary classification to 2-class multiclass output

            Note: Output classes must be located in dimension 1 of y, and for masked aggregation,
            dimension 1 must have a size of 1
            
            :param wrapped_model: the actual Pytroch model to wrap
            :param out_agg_dim: None or tuple, if tuple, output dims wrt which we aggregate the output
            :param num_classes: Number of output classes or features that have been predicted
            :param cross_agg: If True, generate all possible combinations of the output classes with the masking classes
            :param agg_dim_fn: pytorch function used to aggregate `y` if not using masked aggregation. It must accept dim param
            :param agg_mask_fn: pytorch function used to aggregate `y` if using masked aggregation. It must accept axis param
            :param using_mask: Use mask aggregation, instead of aggregation acrross dimensions
            :param is_GP: Whether the model has a Gaussian Process as the final layer
            :param ignore_classes: list of output classes to ignore. If ussing -full (cross-aggregation), this
                list must be in the order [(ouput class 1, custom class 1), (ouput class 1, custom class 2)], etc.
        '''
        super(XAIModelWrapper, self).__init__()
        self.out_agg_dim= out_agg_dim
        self.num_classes= num_classes
        self.cross_agg= cross_agg
        self.agg_dim_fn= agg_dim_fn
        self.agg_mask_fn= agg_mask_fn
        self.wrapped_model= wrapped_model
        self.using_mask= using_mask
        self.is_GP= is_GP
        self.ignore_classes= ignore_classes

    def forward(self, *args):
        '''
            :param args: a list of inputs to the model or. If self.using_mask is true, that plus a mask
        '''
        if self.using_mask:
            x, mask= args[:-1], args[-1]
        else:
            x, mask= args, None
                        
        if self.is_GP:
            out= self.wrapped_model.likelihood(self.wrapped_model(*x)).mean.unsqueeze(-1)
        else:
            out= self.wrapped_model(*x)
            
        #Convert eveything to float32, because otherwise we will get overflows in the agg step
        out, mask= out.to(torch.float32), mask.to(torch.float32)
        
        #Return
        return self.agg_y(out, mask=mask)
        
    def agg_y(self, y:torch.Tensor, mask:Optional[torch.Tensor]=None, eps=1e-5):
        '''
            :param y: Model predictions or labels to be aggregated
            :param mask: If provided (not None), mask to aggregate over the classes

            :return: aggregated y
        '''     
        #Get shapes
        y_shape= tuple(y.shape)
        
        #Expand y and mask if there is a single output class, but there should be two
        if y.shape[1]==1 and self.num_classes==2:
            y= torch.concatenate([1-y, y], axis=1)

        #Mask or maskless aggregation
        if mask is not None: #Mask aggregation. If mask is provided, out_agg_dim is ignored
            if mask.shape[1]==1 and self.num_classes==2:
                 mask= torch.concatenate([~mask, mask], axis=1)
            # assert y_shape[0] == 1, f'For now, first dimension of y (batch dim) must be 1. Found: {y_shape[0]}'
            if self.cross_agg:
                new_classes_arr= [torch.sum((y[:,[c1]] * mask[:,[c2]]).reshape(y_shape[0],-1), axis=-1, keepdims=True) /
                        (torch.sum(mask[:,[c2]]) + eps) for c1 in range(y_shape[1]) for c2 in range(mask.shape[1]) 
                        if c1 not in self.ignore_classes]
                y= torch.concatenate(new_classes_arr, axis=1)
            else:
                new_classes_arr= [torch.sum((y[:,[c1]] * mask[:,[c1]]).reshape(y_shape[0],-1), axis=-1, keepdims=True) / 
                        (torch.sum(mask[:,[c1]]) + eps) for c1 in range(y_shape[1]) if c1 not in self.ignore_classes]
                y= torch.concatenate(new_classes_arr, axis=1)
        elif self.out_agg_dim is not None: #Maskless aggregation. 
            y= self.agg_dim_fn(y, dim=self.out_agg_dim, keepdim=False)
        else: pass #No aggregation

        return y