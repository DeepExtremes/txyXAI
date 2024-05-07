import os, sys
from pathlib import Path

import torch
import torchmetrics
import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint, EarlyStopping

#Custom imports
from .model_xytmodel import xytConvLSTM, xytConvTransformer
from .data_loader import DeepExtremes

class ModelManager():
    def __init__(self, config):
        self.config = config
                
        # Build paths
        self.cp_path= os.path.join(self.config['save_path'], 'checkpoints')
        self.log_path= os.path.join(self.config['save_path'])
        
        # Get model
        self.model_class= eval(self.config['arch'].get('model', 'xytConvLSTM'))
    
    def load_data(self, shuffle_all=False, sample=None, **kwargs):
        # Datasets
        data_train= eval(self.config['data']['name'])(self.config, subset='train', **kwargs)
        data_val= eval(self.config['data']['name'])(self.config, subset='val', **kwargs)
        data_test = eval(self.config['data']['name'])(self.config, subset='test', **kwargs)
        
        #Sample datasets?
        data_train_sampler, data_val_sampler, data_test_sampler= None, None, None
        g_cpu = torch.Generator()
        g_cpu.manual_seed(self.config['seed'])
        if not isinstance(sample, list): sample= [sample]*3
        assert len(sample) == 3
        if sample[0] is not None:
            data_train_sampler= RandomSampler(data_train, replacement=False, num_samples=sample[0], generator=g_cpu)
        if sample[1] is not None:
            data_val_sampler= RandomSampler(data_val, replacement=False, num_samples=sample[1], generator=g_cpu)
        if sample[2] is not None:
            data_test_sampler= RandomSampler(data_test, replacement=False, num_samples=sample[2], generator=g_cpu)
        
        # Dataloaders
        self.train_loader = DataLoader(data_train, shuffle=data_train_sampler is None, sampler=data_train_sampler,
                                  batch_size=self.config['implementation']['trainer']['train_batch_size'], 
                                  num_workers=self.config['implementation']['data_loader']['num_workers'])
        
        self.val_loader = DataLoader(data_val, shuffle=shuffle_all, sampler=data_val_sampler,
                                batch_size=self.config['implementation']['trainer']['val_batch_size'],
                                num_workers=self.config['implementation']['data_loader']['num_workers'])
        
        self.test_loader = DataLoader(data_test, shuffle=shuffle_all, sampler=data_test_sampler,
                                 batch_size=self.config['implementation']['trainer']['test_batch_size'],
                                 num_workers=self.config['implementation']['data_loader']['num_workers'])

    def implement_model(self, naive=None):
        # Model
        self.model = self.model_class(self.config, naive=naive)
        #self.model.to(torch.float32)  

        
        checkpoint_callback = ModelCheckpoint(dirpath = self.cp_path,
           filename = '{epoch}-{'+self.config['implementation']['trainer']['monitor']['split']+\
                                              '_'+self.config['implementation']['trainer']['monitor']['metric']+':.6f}',
           mode = self.config['implementation']['trainer']['monitor_mode'],
           monitor =  self.config['implementation']['trainer']['monitor']['split']+\
                                              '_'+self.config['implementation']['trainer']['monitor']['metric'],
           save_last = True, save_top_k = 30)
        
        early_stopping = EarlyStopping(
           monitor = self.config['implementation']['trainer']['monitor']['split'] + '_' +\
                     self.config['implementation']['trainer']['monitor']['metric'],
           min_delta = 0.0, 
           patience = self.config['implementation']['trainer']['early_stop'], 
           verbose = False, mode = self.config['implementation']['trainer']['monitor_mode'], strict = True)
        
        callbacks = [checkpoint_callback, early_stopping, ModelSummary(max_depth=-1)]
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.log_path)

        # Trainer
        self.trainer = pl.Trainer(accumulate_grad_batches=self.config['implementation']['trainer']['accumulate_grad_batches'], 
                                  callbacks = callbacks, 
                                  accelerator=self.config['implementation']['trainer']['accelerator'],
                                  devices=self.config['implementation']['trainer']['devices'],
                                  # gradient_clip_val = self.config['implementation']['optimizer']['gclip_value'],
                                  logger = [tb_logger], 
                                  max_epochs = self.config['implementation']['trainer']['epochs'],
                                  precision=self.config['implementation']['trainer']['precission'],
                                  reload_dataloaders_every_n_epochs = 1, 
                                  val_check_interval = 1.0, 
                                  enable_model_summary=True,
                                  log_every_n_steps=10,
                                  # limit_train_batches=2, limit_val_batches=2, log_every_n_steps=1, #Only for debugging
                                 ) 
    def train(self):
        self.trainer.fit(self.model, self.train_loader, self.val_loader) 
        return self.model
    
    def test(self, naive=None, blind_test=False, event_metrics=None):
        self.val(subset='test', naive=naive, blind_test=blind_test, event_metrics=event_metrics)
        
    def val(self, subset='val', plot=6, naive=None, blind_test=False, event_metrics=None):
        '''
            naive: one of ['climatology', 'last-value', None]
            blind_test: True/False or a list of int (timesteps) or a slice over time dimension
            event_metrics ['none' (evaluate over no event timesteps), 'all' (evaluate over all event timesteps),
                           int (a over a specfic event), None (evaluate all events and non-events)]
        '''
        #Pass through the naive option
        self.model.naive= naive
        
        #Pass through the blind test option
        self.model.blind_test= blind_test
        
        #Pass through the event metrics option
        self.model.event_metrics= event_metrics
        
        #Get dataloader
        assert subset in ['train', 'val', 'test']
        loader= {'train':self.train_loader, 'val':self.val_loader, 'test':self.test_loader}[subset]
        
        #Set up plotting 6 batches, and to save plots
        print(f'Plotting first {plot} batches')
        self.model.save_path, self.model.plot_every_epoch= Path(self.config['save_path']) / f'{subset}_plots', plot
        
        #Validate
        self.model.eval()
        with torch.no_grad():
            self.trainer.validate(model=self.model, dataloaders=loader)
        
        #Reset plotting to defaults (just one plot per epoch, and not save plots)
        self.model.save_path, self.model.plot_every_epoch= None, 1
        
    def inference(self, subset='test'):
        # Inference 
        assert subset in ['train', 'val', 'test']
        loader= {'train':self.train_loader, 'val':self.val_loader, 'test':self.test_loader}[subset]
        evaluator = PytorchEvaluator(self.config, self.model, loader)

        inference_outputs = evaluator.inference()
        evaluator.evaluate(inference_outputs)
        
        return inference_outputs
    
    def load_custom(self, ckpt=None):
        path= Path(self.cp_path) / ckpt
        self.model= self.model_class.load_from_checkpoint(
            checkpoint_path=path, config=self.config)
        #self.model.to(torch.float32)  
        print(f'Loaded model from {path}')
        return self.model
    
    def load_last(self):
        #Find last model
        last_candidates= [f for f in os.listdir(self.cp_path) if 'last' in f]
        if len(last_candidates) == 1:
            last_name= last_candidates[0]
        elif len(last_candidates) > 1:
            last_name= sorted(last_candidates, key=lambda filename: 
                              int(filename.split('v')[-1].replace('.ckpt', '')) if '-v' in filename else -100)[-1]
        else:
            raise AssertionError(f'No last.ckpt found in {self.cp_path}')

        last_model= os.path.join(self.cp_path, last_name)
        loaded_model= self.model_class.load_from_checkpoint(checkpoint_path=last_model, config=self.config)
        
        #Since we are still making many changes, let's just load the weights of the model and nothing more
        #self.model= loaded_model
        self.model= loaded_model
        # self.model.rnn= loaded_model.rnn
        # self.model.mlp= loaded_model.mlp
        #self.model.to(torch.float32)   
        
        print(f'Loaded model from {last_model}')
        return self.model
    
    def load_best(self):
        # Select best model
        list_checkpoints = [filename for filename in os.listdir(self.cp_path)
                            if self.config['implementation']['trainer']['monitor']['split']+
                            '_'+self.config['implementation']['trainer']['monitor']['metric'] in filename
                            ]
        assert len(list_checkpoints), f'Checkpoint list empty at {self.cp_path}'
        best_model= sorted(list_checkpoints, key=lambda filename: 
                               float(filename.split('=')[-1].replace('.ckpt', '')) if '=' in filename else -1e6)[-1]
        self.config['best_run_path']= os.path.join(self.cp_path, best_model)
        loaded_model= self.model_class.load_from_checkpoint(checkpoint_path=self.config['best_run_path'], config=self.config)
        
        #Since we are still making many changes, let's just load the weights of the model and nothing more
        #self.model= loaded_model
        self.model= loaded_model
        # self.model.rnn= loaded_model.rnn
        # self.model.mlp= loaded_model.mlp
        #self.model.to(torch.float32)   
        
        print(f'Loaded model from {self.config["best_run_path"]}')
        return self.model