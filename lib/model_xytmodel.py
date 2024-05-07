import numpy as np
import os, sys, inspect, abc
from pathlib import Path
from itertools import chain

import torch
import torchmetrics
from torchmetrics import Metric
import pytorch_lightning as pl
from torchvision.ops import MLP

#Custom imports
from .model_convLSTM import ConvLSTMCustom
from .model_transformer import ConvTransformerCustom
from .data_utils import ndvi as _ndvi, NDVI, plot_prediction, prepare_batch, get_nth
    
class xytModel(pl.LightningModule, metaclass=abc.ABCMeta):   
    def __init__(self, config, naive=None, blind_test=False, event_metrics=None):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()

        # Load Config
        self.config = config
        
        self.use_txy_mask_as_input= config['data']['variables']['use_txy_mask_as_input']
        
        self.input_dimension= config['arch']['input_dimension']

        self.num_classes= config['arch']['num_classes']
        self.pred_period= config['data'].get('prediction_period', 1) #Number of timesteps to predict into the future

        self.not_apply_metrics_to_channels= config['evaluation'].get('not_apply_metrics_to_channels', None)
        if self.not_apply_metrics_to_channels is not None:
            print(f'Warning: metrics are not being applied for channels {self.not_apply_metrics_to_channels}')
        self.difference_to_clima= config['arch'].get('difference_to_clima', True)
        self.test_start_idx= None
        self.keep_inputs_txy= config['arch'].get('keep_inputs_txy', None)
        if self.keep_inputs_txy is not None:
            print(f'Warning: inputs {self.keep_inputs_txy} will be removed from the input')
        
        self.plot_every_epoch= config['implementation']['trainer']['plot_every_epoch']
        self.plotted= 0 #Used to keep track of the number of plotted cases per epoch
        self.save_path= Path(self.config['save_path']) / 'train_plots'
                
        # Loss
        self.loss= set_loss(self.config['implementation']['loss'])

        # Initialize Logger Variables
        self.loss_train = []
        self.loss_val = []
        self.loss_test = []

        # Initialize Test Metrics
        self.train_metrics = init_metrics(self.config['evaluation']['metrics'])
        self.val_metrics = init_metrics(self.config['evaluation']['metrics'])
        self.test_metrics = init_metrics(self.config['evaluation']['metrics'])
        
        #If naive is on, use a naive non-trained model instead
        self.naive= naive
        
        #If blind test is True, the last year of reflectances is set to 0
        #It can also be a slice or a list of timesteps that we want to set to zero
        self.blind_test= blind_test
        
        #Apply metrics only over event / non-event areas / specific event areas
        self.event_metrics= event_metrics
        
        #This idx are used for naive models and for self.difference_to_clima.
        self.band_names= config['implementation']['band_names']
        self.output_band_idx= None #[0,1,2,3] #rgb-nir
        self.band_idx= None #self.output_band_idx #rgb-nir
        
        self.clima_names= config['implementation']['clima_names']
        self.clima_idx= None #[4,5,6,7] #[-5,-4,-3,-2] #rgb_clima-nir_clima
        
        self.cloud_mask_name= config['implementation']['cloud_mask_name']
        self.cloud_mask_idx= None #-1 #cloud mask
        
        #If not None, set a continuous set of reflectances to zero during training strating from a random 
        #timestep and lasting with a random amount of time between (low, high)
        self.random_zeroing_duration= config['arch'].get('random_zeroing_duration', None)
        
    @abc.abstractmethod
    def _forward(self, t, xy, txy, *args, **kwargs):
        pass
    
    def get_train_mask(self, mask):
        return mask
    
    def get_txy_params(self):
        'Returns parameters for which the xyt dimensions are NOT stacked onto the batch dimension'
        return []

    def get_xy_params(self):
        'Returns parameters for which the xy dimensions are NOT stacked onto the batch dimension (but t is)'
        return []
    
    def get_t_params(self):
        'Returns parameters for which the t dimensions are NOT stacked onto the batch dimension (but xy are)'
        return []
    
    def get__params(self):
        'Returns parameters for all dimensions (txy) are stacked onto the batch dimension'
        return []
        
    def forward(self, t, xy, txy, *args, train_mask=None, **kwargs):
        #Unpack x, make sure all tensors are in the correct device
        t, xy, txy= t.to(self.device), xy.to(self.device), txy.to(self.device)
        
        #Keep only some inputs for the model
        if self.keep_inputs_txy is not None:
            txy= txy[:,self.keep_inputs_txy]
        
        #Forward
        if self.naive is None:
            if self.blind_test is not None and \
                    ( isinstance(self.blind_test, bool) and self.blind_test or \
                      not isinstance(self.blind_test, bool) ): #A list of timesteps or a slice
                txy_fw= txy.clone() #b c t h w 
                if isinstance(self.blind_test, bool):
                    txy_fw[:,:,self.test_start_idx:]= 0. #Set bands to 0
                    txy_fw[:,self.cloud_mask_idx, self.test_start_idx:]= 1. #Set mask to 1
                else:
                    txy_fw[:,:,self.blind_test]= 0. #Set bands to 0
                    txy_fw[:,self.cloud_mask_idx, self.blind_test]= 1. #Set mask to 1
            elif self.random_zeroing_duration is not None:
                low, high= self.random_zeroing_duration
                duration= torch.randint(low=low, high=high, size=(1,))[0]
                start= torch.randint(low=0, high=txy.shape[2], size=(1,))[0]
                end= min(start + duration, txy.shape[2])
                
                txy_fw= txy.clone() #b c t h w 
                txy_fw[:,:, start:end]= 0. #Set bands to 0
                txy_fw[:, self.cloud_mask_idx, start:end]= 1. #Set mask to 1
            else:
                txy_fw= txy
            output= self._forward(t, xy, txy_fw, *args, train_mask=train_mask, **kwargs)
            
            #Return
            if self.difference_to_clima:
                output= output + txy[:,self.clima_idx]
            
        else: #Use naive model
            if self.naive=='climatology': #Use monthly climatology for ndvi (the other outputs are set to 0)
                output= txy[:,self.clima_idx] #-1 is cloud mask, and -2 is clima_ndvi
                # output[:,-1]= torch.sqrt(torch.atanh(txy[:,-2])) #If using kndvi
            elif self.naive=='last-value': #Use previously availale non-cloudy value
                out_shape= list(txy.shape)
                out_shape[1]= self.num_classes
                output= torch.zeros(out_shape).to(self.device)
                for i in range(output.shape[2]): #iterate over time channel
                    #If there is cloud, use previous value, otherwise, use input values
                    output[:,self.output_band_idx,i]= torch.where(txy[:,[self.cloud_mask_idx],i] > 0., 
                           output[:,self.output_band_idx,i-1] if i!=0 else 0., txy[:,self.band_idx,i])
            else:
                valid_values= ['climatology', 'last-value']
                raise AssertionError(f'{self.naive=} must be one of {valid_values=}')

        #Add ndvi and return
        output= torch.cat([output, _ndvi(output[:,[2]], output[:,[3]])], axis=1)
        return output

    def shared_step(self, batch, mode):
        #Get batch
        x, labels, masks, metric_masks, t_labels, t_dates_str, t, xy, txy= \
            prepare_batch(batch, self.num_classes, self.pred_period, self.device)
        
        #Get proper test_start_idx
        if self.test_start_idx is None:
            self.test_start_idx= batch['meta']['test_start_idx'][0]
            if self.blind_test:
                print(f' - Warning: using {self.blind_test=}')
            if self.random_zeroing_duration is not None:
                print(f' - Warning: using {self.random_zeroing_duration=}')
        
        #Get band indices from dataloader meta information
        if self.output_band_idx is None:
            self.band_idx= [get_nth(batch['meta']['names']['txy']).index(b) for b in self.band_names]
            self.output_band_idx= self.band_idx
            try:
                self.clima_idx= [get_nth(batch['meta']['names']['txy']).index(b) for b in self.clima_names]
            except Exception as e:
                print(f'Error: {e}')
                self.clima_idx= None
            self.cloud_mask_idx= get_nth(batch['meta']['names']['txy']).index(self.cloud_mask_name)
            print(f'Warning: {self.band_idx=}, {self.output_band_idx=}, {self.clima_idx=}, {self.cloud_mask_idx=}')

        #Get a random start point (we lose the beginning chunk_size / 2) timespteps for training
        if hasattr(self, 'chunk_size') and self.chunk_size is not None and self.training:
            raise NotImplementedError('`chunk_size` is now deprecated. Do not activate!')
            st= torch.randint(low=0, high=self.chunk_size, size=(1,))[0]
            x= (x[0][:,:,st:], x[1], x[2][:,:,st:])
            labels, masks= labels[:,:,st:], masks[:,:,st:]
            t_labels, t_dates_str= t_labels[:,st:], t_dates_str[st:] 
            #t, xy, txy= t[st:], xy[st:], txy[st:] #TODO
        
        #Go
        masks= self.get_train_mask(masks)
        output= self(*x, train_mask=masks)
                
        # Compute loss
        if 'weight' in self.config['implementation']['loss']['params'].keys() and \
            not isinstance(self.config['implementation']['loss']['params']['weight'], torch.Tensor):
                self.config['implementation']['loss']['params']['weight'] =\
                    torch.Tensor(self.config['implementation']['loss']['params']['weight'])

        #Get rid of the ndvi calculated in forward
        # output= output[:,:labels.shape[1]]
        
        #Or just add it to the labels and masks
        labels= torch.cat([labels, _ndvi(labels[:,[2]], labels[:,[3]])], axis=1)
        masks= torch.cat([masks, masks[:,[0]]], axis=1) #Extend mask one more channel
        
        #Loss
        loss= self.loss(output, labels) 
        loss= loss[masks].mean()
                   
        #Log
        self.log_dict({f'{mode}_loss':loss}, batch_size=x[0].shape[0], prog_bar=True)
        self.step_metrics(output.detach(), labels, txy_mask=masks, xy_mask=metric_masks, t_mask=t_labels, mode=mode)
        
        #Plot?
        if (isinstance(self.plot_every_epoch, int) and self.plot_every_epoch > self.plotted and mode=='val') or\
           (isinstance(self.plot_every_epoch, list) and any([Path(id).stem in self.plot_every_epoch 
                                                             for id in batch['meta']['id']])): 
            #Pass detached numpy tensors to not break anything
            plot_prediction(
                 x[-1].detach().cpu().numpy(), masks.detach().cpu().numpy(), 
                 labels.detach().cpu().numpy(), output.detach().cpu().numpy(), 
                 t_dates_str[self.pred_period:], 
                 t[:,:,self.pred_period:].detach().cpu().numpy(), 
                 t_labels[:,0].detach().cpu().numpy(),
                 get_nth(batch['meta']['names']['t']), save_path=self.save_path, 
                 select=np.s_[150:200:1] if self.save_path is None else np.s_[::1],
                 title=f'{mode=} epoch={self.current_epoch} loss={loss.detach().cpu().numpy():.5f}',
                 ids=batch['meta']['id'], N='all' if mode != 'train' else 5, naive=self.naive)
            self.plotted+= 1
            #torch.cuda.empty_cache()
        
        return {'loss': loss, 'output': output.detach(), 'labels': labels}

    def step_metrics(self, outputs, labels, txy_mask=None, xy_mask=None, t_mask=None, mode='train') :
        'Update evaluation metrics'
        for metric_name, metric in getattr(self, mode+'_metrics').items():
            mmask= self.get_metric_mask(outputs, txy_mask, xy_mask, t_mask, mode)
            metric.to(self.device).update(outputs[mmask], labels[mmask])

    def get_metric_mask(self, outputs, txy_mask=None, xy_mask=None, t_mask=None, mode='test'):
        '''
            Mask out anything that we do not want to apply our metrics on
            For now, we are masking out all channels but NDVI
            Also, mask out training timesteps, and evaluate only on validation timesteps
        '''
        #Base mask with all outputs that are not nan
        mmasks= ~outputs.isnan() #torch.ones(outputs.shape, dtype=torch.bool)
        
        # ----- Custom masking logic -----
        
        #Add event mask: it is True for event / non event time periods, depending on configuration
        if t_mask is not None:
            if self.event_metrics is None:
                t_mask= None #If self.event_metrics is None, ignore t_mask!!!
            elif isinstance(self.event_metrics, str): 
                if self.event_metrics == 'all': t_mask= t_mask > 0
                elif self.event_metrics == 'none': t_mask= t_mask == 0
                else: raise AssertionError(f'{self.event_metrics=} must either be `all`, `none`, an int or None')
            elif isinstance(self.event_metrics, int):
                t_mask= t_mask == self.event_metrics
            else:
                raise AssertionError(f'{self.event_metrics=} must either be `all`, `none`, an int or None')
        
        #Apply mask at the channel dimension
        #Do not apply metric over some output channels (e.g. consider only ndvi)
        if self.not_apply_metrics_to_channels is not None:
            mmasks[:, self.not_apply_metrics_to_channels]= False
            
        #If in validation or test, apply metric only for values starting from self.test_start_idx
        #For train, apply to all timesteps. To change this behaviour, we should need to return the last
        #year of data for training batches, which we are not currently doing
        if mode != 'train' and self.test_start_idx > 0 and self.test_start_idx < (mmasks.shape[2] + 1):
            mmasks[:, :, :self.test_start_idx]= False 
                 
        # ----- General masking logic -----
        
        #`txy_mask` is True for valid pixels (non-cloud-covered pixels)
        if txy_mask is not None: mmasks= mmasks & txy_mask.bool()
                    
        #`xy_mask` is True for vegetation pixels
        if xy_mask is not None: mmasks= mmasks & xy_mask[:,:,None].bool()
        
        #`t_mask` is True for event / non event time periods, depending on configuration
        if t_mask is not None: mmasks= mmasks & t_mask[...,None,None].bool()
        
        return mmasks

    def epoch_metrics(self, mode):
        'Compute and log evaluation metrics. Reset metrics'
        # Compute and log average metrics
        batch_size= {'train':self.config['implementation']['trainer']['train_batch_size'],
                     'val':self.config['implementation']['trainer']['val_batch_size'],
                     'test':self.config['implementation']['trainer']['test_batch_size']}[mode]
        visualize_in_prog_bar = [self.config['implementation']['trainer']['monitor']['metric']]
        for metric_name, metric in getattr(self, mode+'_metrics').items():
            metric_values= metric.compute()
            if isinstance(metric_values, torch.Tensor): metric_values={'0':metric_values}
            self.log_dict({f'{mode}_{metric_name}_{submetric}':v for submetric, v in metric_values.items()}, 
                     prog_bar=metric_name in visualize_in_prog_bar, batch_size=batch_size)

        # Reset metrics
        for metric_name, metric in getattr(self, mode+'_metrics').items():
            metric.reset()

    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, mode = 'train')
        self.loss_train.append(res['loss'])
        return {'loss': res['loss'], 'output': res['output'], 'labels': res['labels']}

    def on_train_epoch_end(self):
        self.plotted= 0
        self.epoch_metrics(mode = 'train')
    
    def validation_step(self, batch, batch_idx): 
        res = self.shared_step(batch, mode = 'val')
        self.loss_val.append(res['loss'])
        return {'val_loss': res['loss'], 'output': res['output'], 'labels': res['labels']}

    def on_validation_epoch_end(self):
        self.plotted= 0
        self.epoch_metrics(mode = 'val')
    
    def test_step(self, batch, batch_idx):
        res = self.shared_step(batch, mode = 'test')        
        return {'test_loss': res['loss'], 'output': res['output'], 'labels': res['labels']}

    def on_test_epoch_end(self):
        self.epoch_metrics(mode = 'test')
        
    def configure_optimizers(self):
        """
        Configure optimizer for training stage
        """
        # build optimizer
        lr= self.config['implementation']['optimizer']['lr']
        final_lr= self.config['implementation']['optimizer'].get('final_lr', None)
        epochs= self.config['implementation']['trainer']['epochs']
        
        t_p, xy_p, txy_p, _p= self.get_t_params(), self.get_xy_params(), self.get_txy_params(), self.get__params()
        lr_correction= self.config['implementation']['optimizer'].get('lr_correction',{})
        if _p == [] and t_p == [] and xy_p == [] or not len(lr_correction):
            trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        else:
            assert hasattr(self.input_dimension, '__len__') and len(self.input_dimension) == 3
            c_t, c_xy, x_txy= self.input_dimension #Get number of channels
            txy_lr, xy_lr= lr*lr_correction.get('txy', 1), lr*lr_correction.get('xy', 1)
            t_lr, _lr= lr*lr_correction.get('t', 1), lr*lr_correction.get('_', 1)
            trainable_params= [] +  ([{'params':_p, 'lr':_lr}] if _p!=[] else []) +\
                                    ([{'params':t_p, 'lr':t_lr}] if t_p!=[] else []) +\
                                    ([{'params':xy_p, 'lr':xy_lr}] if xy_p!=[] else []) +\
                                    ([{'params':txy_p, 'lr':txy_lr}] if txy_p!=[] else [])
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
        
        if final_lr is None or final_lr == lr:
            return optimizer
        else:
            print(f'Warning: using linear LR scheduling from {lr=} to {final_lr=}')
            lr_scheduler= torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., 
                                    end_factor=final_lr/lr, total_iters=epochs, last_epoch=-1, verbose=True)
            return {"optimizer": optimizer, "lr_scheduler":lr_scheduler}
     
#Define the actual models
class xytConvTransformer(xytModel):
    def __init__(self, config, naive=None):
        super().__init__(config, naive=naive)
        
        #Extract model-specific parameters
        tf_params= dict(n_heads=4, n_layers=3, n_in_tf=64, n_hidden=64,
                        convs=2, dilation=(4,4), kernel_size=(3,3), dropout=0.1)
        for k, v in config['arch']['transformer_params'].items():
            tf_params[k]= v
                
        # Define model
        self.xyt_conv_tf= ConvTransformerCustom(self.input_dimension, self.num_classes, **tf_params)

    def _forward(self, t, xy, txy, *args, train_mask=None, **kwargs):
        return self.xyt_conv_tf(t, xy, txy, train_mask)
            
    def get_xy_params(self):
        return self.xyt_conv_tf.get_xy_params()
    
    def get_t_params(self):
        return self.xyt_conv_tf.get_t_params()

class xytConvLSTM(xytModel):
    def __init__(self, config, naive=None):
        super().__init__(config, naive=naive)
        
        #Extract model-specific parameters
        self.lstm_hidden_size= config['arch']['lstm_hidden_size']
        self.lstm_kernel_size= config['arch']['lstm_kernel_size']
        self.lstm_num_layers= config['arch']['lstm_num_layers']
        self.lstm_convs_per_layer= config['arch'].get('lstm_convs_per_layer', 1)
        self.lstm_dilation= config['arch'].get('dilation', 1)
        self.chunk_size= config['arch'].get('chunk_size', None) #Split training into time chunks to improve speed
        self.mlp_layer_sizes= config['arch']['mlp_layer_sizes']
        self.residual= config['arch'].get('residual', False)
        self.no_memory= config['arch'].get('no_memory', False)
                
        # Define model
        self.rnn= ConvLSTMCustom(
             input_dim=sum(self.input_dimension), hidden_dim=self.lstm_hidden_size, kernel_size=self.lstm_kernel_size, 
             num_layers=self.lstm_num_layers, batch_first=True, bias=True, return_all_layers=False, 
             convs_per_layer=self.lstm_convs_per_layer, dilation=self.lstm_dilation,
            residual=self.residual, no_memory=self.no_memory)#.half()#, n_groups=4)
        self.mlp= MLP(in_channels=self.lstm_hidden_size, hidden_channels=[*self.mlp_layer_sizes, self.num_classes], 
                      dropout=0., inplace=False)
    
    def get_txy_params(self):
        return self.rnn.parameters()
    
    def get__params(self):
        return self.mlp.parameters()
    
    def _forward(self, t, xy, txy, *args, **kwargs):
        #Permute where needed
        txyp= txy.permute(0,2,1,3,4) #b c t h w -> b t c h w
        xyp= xy.permute(0,1,2,3) #b c h w -> b c h w
        tp= t.permute(0,2,1) #b c t -> b t c

        #( (b c h w), (b t c h w), (b t c) )
        layer_output, layer_state= self.rnn((txyp, xyp, tp), chunk_size=self.chunk_size if self.training else None) 
        layer_output_last= layer_output[-1] #Extract last layer's output
        #[494, 1, 20, 128, 128] t b c h w -> bthw c(20)
        output= self.mlp(layer_output_last.permute(1,0,3,4,2).reshape(-1, self.lstm_hidden_size))
        #x.shape: b c t h w | Then: bthw c(4) -> b t h w c -> b c t h w
        output= output.reshape(txy.shape[0], layer_output_last.shape[0], txy.shape[3], 
                               txy.shape[4], self.num_classes).permute(0,4,1,2,3) 
        
        return output        
    
#Metrics and losses
#TODO: For whatever reason, defining the metrics externally breaks everything
# from .model_metrics import L1L2Loss, NormalizedNashSutcliffeEfficiency_custom, L1_custom
class NormalizedNashSutcliffeEfficiency_custom(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("nse_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("nnse_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("bias_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_obs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target) -> None: 
        if len(preds):
            #Standard squared implementation
            se= (preds - target)**2
            var= (target - target.mean())**2
            nse = 1 - se.sum() / (var.sum() + 1e-6)
            nnse = 1 / (2 - nse)

            #Also compute bias
            bias= torch.abs(preds.mean() - target.mean()).mean()

            #Update states
            self.nse_sum+= nse
            self.nnse_sum+= nnse
            self.bias_sum+= bias
            self.n+= 1
            self.n_obs+= len(preds)

    def compute(self):
        return {'NNSE': self.nnse_sum.sum() / (self.n + 1e-6),
                'NSE': self.nse_sum.sum() / (self.n + 1e-6),
                'bias': self.bias_sum.sum() / (self.n + 1e-6)}
    
class L1_custom(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("abs_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target) -> None:
        if len(preds):
            self.abs_sum+= torch.abs(preds - target).sum()
            self.n+= len(preds)

    def compute(self):
        return self.abs_sum.sum() / (self.n + 1e-6)

class L1L2Loss(torch.nn.Module):
    def __init__(self, lam:float=0.9, weights=None, channel_dim=1, ndims=5, **kwargs):
        super().__init__()
        self.lam= lam
        self.l1= torch.nn.L1Loss(**kwargs)
        self.l2= torch.nn.MSELoss(**kwargs)
        self.c= channel_dim
        
        #Build a weights matrix with appropiately placed singleton dimensions
        if weights is None:
            self.weights= None
        else:
            self.weights= torch.Tensor(weights)
            for dim in range(ndims):
                if dim < channel_dim: self.weights= self.weights[None]
                elif dim > channel_dim: self.weights= self.weights[...,None]
                else: pass #Do nothing when the current dim is the channel dim
                
    def forward(self, x, y):
        loss= self.lam*self.l1(x, y) + (1.-self.lam)*self.l2(x, y)
        if self.weights is not None: 
            loss= loss * self.weights.to(loss.device)
        return loss


def init_metrics(metrics_list):
    metrics_dict = {}
    for metric_name, metric_params in metrics_list.items():
        try:
            metrics_dict[metric_name] = getattr(torchmetrics, metric_name)(**metric_params)
        except:
            metrics_dict[metric_name] = eval(metric_name)(**metric_params)
    return metrics_dict

class Loss(torch.nn.Module):
    """ This class is used for wrapping for Torch loss functions """
    def __init__(self, package, configuration):
        super().__init__()
        self.configuration = configuration
        self.loss = getattr(package, configuration['type'])
    
    def forward(self, outputs, labels):
        return self.loss(outputs, labels, **self.configuration['params'])

def set_loss(parameters):
    # Check if loss is user defined
    if not parameters['user_defined']:
        # Import python package containig the loss
        package = __import__(parameters['package'], fromlist=[''])

        # Check if chosen loss is a class or a function module
        if inspect.isclass(getattr(package, parameters['type'])):
            loss = getattr(package, parameters['type'])(**parameters['params'])
        else:
            # If function, create class wrapper 
            loss = Loss(package, parameters)
        
    else:
        # Create user defined loss class
        loss = globals()[parameters['type']](**parameters['params'])

    return loss