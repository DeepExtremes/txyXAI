# import xarray as xr
# import numpy as np
# import fsspec 
import pandas as pd
# import matplotlib.pyplot as plt
# import random, tqdm
#import geopandas as gpd

import json, sys
from pathlib import Path
#from lovely_numpy import lo
import requests
import datetime

import subprocess
from pathlib import Path

import subprocess
from pathlib import Path

'''
Most functions here require awscli with your credentials set up.
Please make sure to do so:
    conda install -c conda-forge awscli
    aws iam create-login-profile --user-name XXXX --password XXXX
'''

BUCKET= 's3://deepextremes-minicubes/'

def aws_sync(aws_path:str, local_path:str, verbose:str=False):
    'Lightweight wrapper around aws sync'
    return aws_transfer(aws_path, local_path, verbose=verbose, sync=True)

def aws_cp(aws_path:str, local_path:str, verbose:str=False):
    'Lightweight wrapper around aws cp'
    return aws_transfer(aws_path, local_path, verbose=verbose, sync=False)

def aws_transfer(aws_path:str, local_path:str, verbose:str=False, sync=False):
    'Lightweight wrapper around aws cp and sync'
    #Remove initial and final /
    #If sync == True, use sync instead of cp
    if aws_path.startswith('/'): aws_path= aws_path[1:]
    if not sync:
        if aws_path.endswith('/'): aws_path= aws_path[:-1]
    else:
        if not aws_path.endswith('/'): aws_path= aws_path+'/'
    
    #Makedir
    if not sync:
        Path(local_path).parent.mkdir(exist_ok=True, parents=True)
    else:
        Path(local_path).mkdir(exist_ok=True, parents=True)
    
    #Call awscli
    out= subprocess.run(['aws', 's3', 'cp' if not sync else 'sync', 
                         BUCKET + aws_path, local_path],
                         text=True, capture_output=True)
    if out.returncode:
        print(f'{out.returncode=}')
        print(out)
    return out.stdout if verbose else None

def aws_get_cube_registry(name:str='mc_registry_v3.csv',
                          path:Path=Path('/data/deepex/'), 
                          verbose:bool=True):
    '''Get the minicube registry .csv from the deepextremes repo'''
    local_name= datetime.datetime.now().strftime('registry_%Y_%m_%d_%H_%M_%S.csv')
    out= aws_cp(name, str(path / local_name), verbose=verbose)
    if verbose: print(out)
    registry_df= pd.read_csv(str(path / local_name))
    registry_df= registry_df.set_index('location_id')
    return registry_df 

def aws_get_versions(name:str='COMPONENTVERSIONS.md',
                     path:Path=Path('/data/deepex/'), 
                     verbose:bool=True):
    '''Get the minicube COMPONENTVERSIONS.md from the deepextremes repo'''
    out= aws_cp(name, str(path / name), verbose=verbose)
    if verbose: print(out)
    return open(str(path / name), 'rt').read()

def aws_ls(path:str='', recursive:bool=False):
    'Lightweight wrapper around aws ls. Returns a list [date str, size, file name]'
    #Remove initial /, Add final /
    if path.startswith('/'): path= path[1:] 
    if not path.endswith('/') and len(path): path+= '/'
    
    #Call awscli
    out= subprocess.run(['aws', 's3', 'ls', BUCKET + path, 
                        ] + (['--recursive'] if recursive else []) , 
                        text=True, capture_output=True)
    if out.returncode: 
        print(f'{out.returncode=}')
        print(out)
        
    #Process data
    aws_ls= [item.split(' ') for item in out.stdout.split('\n')] #Get output as a list
    aws_ls= [[f'{i[0]} {i[1]}', i[-2], i[-1]] for i in aws_ls if len(i)>6] #Keep date + time
    
    return aws_ls
