# encoding: utf-8
"""
Description:
@author: Dong Shuai
@contact: dongshuai@zsc.edu.cn
"""

import torch
from HPPO2026.Models.Trainer import WTENV
from HPPO2026.Models.WindTurbine import HPPOController, ClassicController,BangBangController,OLCController
import datetime
import shutil
import yaml
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    cfgfile = 'FCN_VEL.yml'
    f = open(rf'./config/{cfgfile}', 'r', encoding='utf8')
    config = yaml.safe_load(f)
    f.close()

    now = datetime.datetime.now()
    outpath = config[
                  'OUTPATH'] + f'\\Infer_{now.year}{now.month:02d}{now.day:02d}_{now.hour:02d}_{now.minute:02d}_{now.second:02d}_' + \
              cfgfile.split('.')[0]
    print(f'All data would be saved in {outpath}')
    os.mkdir(outpath)
    shutil.copy(rf'./config/{cfgfile}',os.path.join(outpath,cfgfile))
    shutil.copy(config['WINDTURBINE'], os.path.join(outpath, config['WINDTURBINE'].split('\\')[-1]))

    for i_source in range(1,5): # 4 source domains
        for i_target in range(1,5): #5 target domains
            config["Net"]["Pretrain"] = fr'bestmodel_domain{i_source}.pth'
            config["TESTDATA"]["FILE_NAME"] = fr'testset_domain{i_target}.npz'
            wtenv = WTENV(config,outpath)
            controllers = {
                           # 'condiction-based':ClassicController(wtenv.infer_num_per_group),
                           # 'bangbang':BangBangController(wtenv.infer_num_per_group),
                           'hppo':HPPOController(wtenv.infer_num_per_group,wtenv.model),
                           # 'openloop':OLCController(wtenv.infer_num_per_group)
                           }
            ep_r, ep_e, ep_q, ep_a, ep_b = wtenv.infer(controllers,True,0)

            infer_str = '    ep_r,   ep_e,  ep_q,  ep_a, ep_b'
            # print('    ep_r,   ep_e,  ep_q,  ep_a, ep_b')
            for c_name in controllers.keys():
                infer_str += '\n'
                infer_str += '{}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n'.format(c_name,np.sum(ep_r[c_name]),np.sum(ep_e[c_name]),np.sum(ep_q[c_name]),np.sum(ep_a[c_name]),np.sum(ep_b[c_name]))
            print(config['Net']['Pretrain'])
            print(infer_str)
            print('\n')

            f_infer = open(wtenv.infer_log_path, 'a')
            f_infer.write(f'Domain {i_source} to {i_target}\n')
            f_infer.write(infer_str)
            f_infer.write('=====================================\n')
            f_infer.flush()
            f_infer.close()

if __name__ == '__main__':
    main()
