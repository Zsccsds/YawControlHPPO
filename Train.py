from HPPO2026.Models.Trainer import WTENV
import datetime
import shutil
import yaml
import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Set random seed for reproducible results across multiple libraries

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """
    Main entry point for the training process
    - Loads configuration from YAML file
    - Creates output directory structure
    - Initializes and starts the training environment
    """
    set_seed(seed=42) # set seed

    # Load configuration from YAML file
    cfgfile = 'FCN_VEL.yml'
    f = open(rf'./config/{cfgfile}', 'r', encoding='utf8')
    config = yaml.safe_load(f)
    f.close()

    now = datetime.datetime.now() # get current time
    outpath = config[
                  'OUTPATH'] + f'\\Train_{now.year}{now.month:02d}{now.day:02d}_{now.hour:02d}_{now.minute:02d}_{now.second:02d}_' + \
              cfgfile.split('.')[0]
    print(f'All data would be saved in {outpath}')
    os.mkdir(outpath)

    # Copy config file and wind turbine model to output directory
    shutil.copy(rf'./config/{cfgfile}', os.path.join(outpath, cfgfile))
    shutil.copy(config['WINDTURBINE'], os.path.join(outpath, config['WINDTURBINE'].split('\\')[-1]))
    os.mkdir(outpath+'\\imgs')
    os.mkdir(outpath+'\\inferout_imgs')
    os.mkdir(outpath + '\\models')

    # Start training
    wtenv = WTENV(config,outpath)   # initialize training environment
    wtenv.trainhppo() # start training hppo

if __name__ == '__main__':
    main()