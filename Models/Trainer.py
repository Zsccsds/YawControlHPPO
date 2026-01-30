# encoding: utf-8
"""
Description:
@author: Dong Shuai
@contact: dongshuai@zsc.edu.cn
"""

import torch
import numpy as np

from HPPO2026.Models.HPPONewModels import build_model

from HPPO2026.Models.WindTurbine import WindTurbine, HPPOController
from HPPO2026.utils.GenWind import loadwinddata
from HPPO2026.utils.Visulization import plot4
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WTENV():
    """
    Wind Turbine Environment class for training and inference of HPPO controller
    """
    def __init__(self, config, outpath):
        self.config = config
        self.wa = 0

        self.model, self.memory = build_model(config, device)

        self.train_para = config['TRAIN']
        # self.wamin = config['TRAIN']['WAMIN']
        # self.wamax = config['TRAIN']['WAMAX']
        # self.wamin_ep = config['TRAIN']['WAMIN_EP']
        # self.wamax_ep = config['TRAIN']['WAMAX_EP']

        self.was = config['TRAIN']['WAS']
        self.was_ep = config['TRAIN']['WAS_EP']
        self.reward_weights = config['TRAIN']['REWARD_WEIGHTS']
        self.out_path = outpath
        self.out_img_path = os.path.join(outpath,'imgs')
        self.inferout_img_path = os.path.join(outpath, 'inferout_imgs')
        self.out_models_path = os.path.join(outpath, 'models')
        self.inputdim = config['Net']['StateDim']

        self.max_epsoide = config['TRAIN']['MAX_EPOCH']
        self.img_num = config['INFER']['IMG_NUM']

        self.num_per_group = config['TRAINDATA']['NUM_PER_GROUP']
        self.group_per_epsoide = config['TRAINDATA']['GROUP_PER_EP']

        self.infer_num_per_group = config['TESTDATA']['NUM_PER_GROUP']
        self.infer_group_per_epsoide = config['TESTDATA']['GROUP_PER_EP']
        self.loadwind = loadwinddata
        self.train_datafile = config['TRAINDATA']['FILE_NAME']
        self.test_datafile = config['TESTDATA']['FILE_NAME']
        self.visualplot = plot4
        self.env = WindTurbine(config['WINDTURBINE'],config['Command'], self.reward_weights)

        self.infer_threshold = config['INFER']['THRESHOLD']/90

        self.log_path = self.out_path+'/train_log.txt'
        self.test_log_path = self.out_path + '/test_log.txt'
        self.infer_log_path = self.out_path + '/infer_log.txt'

    def updatewa(self,ep):
        """
        Update wa parameter based on current episode number

        Args:
            ep: Current episode number
        """
        # if ep < self.wamin_ep:
        #     self.wa = self.wamin
        # if ep >= self.wamin_ep and ep < self.wamax_ep:
        #     self.wa = self.wamin+(ep -self.wamin_ep) / (self.wamax_ep-self.wamin_ep)*(self.wamax-self.wamin)
        # if ep >= self.wamax_ep:
        #     self.wa = self.wamax
        if ep in self.was_ep:
            ind = self.was_ep.index(ep)
            self.wa = self.was[ind]


    def trainhppo(self):
        """
        Main training loop for HPPO (Hierarchical Proximal Policy Optimization) controller
        """
        f = open(self.log_path,'a')
        f_test = open(self.test_log_path,'a')
        infer_str = 'Ep       r       e     q     a       b'
        f_test.write(infer_str)
        f_test.write('\n')
        f_test.flush()

        timestep = 0
        total_reward = 0
        for i_epsoide in range(self.max_epsoide):
            self.updatewa(i_epsoide)
            tmpstr ='wa: {}'.format(self.wa)
            print(tmpstr)
            f.write(tmpstr)
            f.flush()

            ep_r = []
            ep_e = []
            ep_q = []
            ep_a = []
            ep_b = []

            dir_gen_train, vel_gen_train = self.loadwind('./data/'+self.train_datafile)

            for i_group in range(self.group_per_epsoide):
                print('eps:{}, group:{}'.format(i_epsoide, i_group))
                # dir_gen,vel_gen = dir_gen_train[i_group*self.num_per_group:(i_group+1)*self.num_per_group],vel_gen_train[i_group*self.num_per_group:(i_group+1)*self.num_per_group]
                sample_indices = random.sample(range(len(dir_gen_train)), self.num_per_group)
                dir_gen = [dir_gen_train[i] for i in sample_indices]
                vel_gen = [vel_gen_train[i] for i in sample_indices]

                state, _ = self.env.reset(dir_gen, vel_gen)  # 得到环境的反馈，现在的状态
                traj_r = []
                traj_q = []
                traj_e = []
                steps = 0
                escape = np.zeros(self.num_per_group)
                action = np.ones(self.num_per_group)
                period = np.zeros(self.num_per_group)
                while True:
                    escape += 1
                    action_c, period_c = self.model.policy_old.train_act(state, self.memory)

                    for ind in range(len(action_c)):
                        if escape[ind]>=period[ind]:
                            period[ind] = period_c[ind] + 1
                            escape[ind] = 0
                            action[ind] = action_c[ind]

                    state, reward, done, info, _ = self.env.step(action,self.wa)
                    self.memory.rewards.append(reward)
                    self.memory.is_terminals.append(np.tile(done, (reward.shape[0])))

                    timestep += 1
                    if timestep % 200 == 0:
                        self.model.update(self.memory)
                        self.memory.clear_memory()
                        timestep = 0
                    total_reward += reward
                    if done:
                        break
                    steps += 1
                    traj_q.append(info[0])   # reward#
                    traj_r.append(reward)
                    traj_e.append(info[1])

                ep_r.extend(np.sum(np.array(traj_r),axis=0))
                ep_e.extend(np.sum(np.array(traj_e),axis=0))
                ep_q.extend(np.sum(np.array(traj_q),axis=0))
                ep_a.extend(np.sum(np.abs(self.env.Action), axis=1))
                ep_b.extend(np.sum(np.abs(self.env.Action[:, :-1] - self.env.Action[:, 1:]), axis=1))

                self.visualplot(self.env, self.out_img_path, i_epsoide, 0, self.img_num)
            out_str = 'Ep: [{}],' .format(i_epsoide)
            out_str += 'R-mean[{:.2f}],'.format(np.mean(ep_r))
            out_str += 'E-mean[{:.2f}],'.format(np.mean(ep_e))
            out_str += 'Q-mean[{:.2f}],'.format(np.mean(ep_q))
            out_str += 'A-mean[{}], '.format(np.mean(ep_a))
            out_str += 'B-mean[{}]\n'.format(np.mean(ep_b))

            print(out_str)
            f.write(out_str)
            f.flush()
            torch.save(self.model.policy.state_dict(),os.path.join(self.out_models_path,'{}.pth'.format(i_epsoide)))
            if (i_epsoide+1)% 1 == 0:
                controllers = {'hppo': HPPOController(self.infer_num_per_group, self.model)}
                ep_r, ep_e, ep_q, ep_a, ep_b = self.infer(controllers, True, i_epsoide)
                for c_name in controllers.keys():
                    f_test.write('\n')
                    infer_str = 'Ep-{},{}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(i_epsoide,
                                                                                      c_name,
                                                                                     np.mean(ep_r[c_name]),
                                                                                     np.mean(ep_e[c_name]),
                                                                                     np.mean(ep_q[c_name]),
                                                                                     np.mean(ep_a[c_name]),
                                                                                     np.mean(ep_b[c_name]))
                    print(infer_str)
                    f_test.write(infer_str)
                    f_test.flush()
        f_test.close()
        f.close()


    def infer(self, controllers,  visualize = False, ep=0):
        """
           Inference/Evaluation function for trained controllers

           Args:
               controllers: Dictionary of controllers to evaluate
               visualize: Whether to save visualization images
               ep: Episode number for naming purposes

           Returns:
               ep_r, ep_e, ep_q, ep_a, ep_b: Evaluation metrics for each controller
           """
        total_reward = 0
        test_dir_gen, test_vel_gen = self.loadwind('./data/'+self.test_datafile)
        ep_r = dict()
        ep_e = dict()
        ep_q = dict()
        ep_a = dict()
        ep_b = dict()
        for ind_cc in range(len(controllers.keys())):
            controller_name = list(controllers.keys())[ind_cc]
            ep_r[controller_name] = []
            ep_e[controller_name] = []
            ep_q[controller_name] = []
            ep_a[controller_name] = []
            ep_b[controller_name] = []

        for i_group_test in range(self.infer_group_per_epsoide):
            print('group:{}'.format(i_group_test))

            dir_gen = test_dir_gen[i_group_test*self.infer_num_per_group:(i_group_test+1)*self.infer_num_per_group]
            vel_gen = test_vel_gen[i_group_test*self.infer_num_per_group:(i_group_test+1)*self.infer_num_per_group]

            for ind_cc in range(len(controllers.keys())):
                controller_name = list(controllers.keys())[ind_cc]
                controller = controllers[controller_name]

                state, _ = self.env.reset(dir_gen, vel_gen)  # 得到环境的反馈，现在的状态
                traj_r = []
                traj_q = []
                traj_e = []
                timestep = 0

                while True:
                    action = controller(state)
                    state, reward, done, info, _ = self.env.step(action,self.wa)
                    total_reward += reward
                    timestep += 1

                    if done:
                        break
                    traj_q.append(info[0])   # reward#
                    traj_r.append(reward)
                    traj_e.append(info[1])
                ep_r[controller_name].extend(np.sum(np.array(traj_r), axis=0))
                ep_e[controller_name].extend(np.sum(np.array(traj_e), axis=0))
                ep_q[controller_name].extend(np.sum(np.array(traj_q), axis=0))
                ep_a[controller_name].extend(np.sum(np.abs(self.env.Action), axis=1))
                ep_b[controller_name].extend(np.sum(np.abs(self.env.Action[:, :-1] - self.env.Action[:, 1:]), axis=1))
                cc_imgpath = os.path.join(self.inferout_img_path,controller_name)
                if not os.path.exists(cc_imgpath):
                    os.makedirs(cc_imgpath)
                if visualize:
                    self.visualplot(self.env, cc_imgpath, ep, i_group_test, self.img_num)
        return ep_r,ep_e,ep_q,ep_a,ep_b

    def infer_hppocontroller_with_record(self, hppo_controller):
        """
        Simulate a trained HPPO controller to collect data for stability analysis

        Args:
            hppo_controller: Trained HPPO controller to simulate

        Returns:
            action_record, e_record, action_c, period_c, Dpsi: Collected data for analysis
        """
        total_reward = 0
        test_dir_gen, test_vel_gen = self.loadwind('./data/'+self.test_datafile)

        action_record = []
        e_record= []
        action_c = []
        period_c = []
        Dpsi = []

        for i_group_test in range(self.infer_group_per_epsoide):
            print('group:{}'.format(i_group_test))

            dir_gen = test_dir_gen[i_group_test*self.infer_num_per_group:(i_group_test+1)*self.infer_num_per_group]
            vel_gen = test_vel_gen[i_group_test*self.infer_num_per_group:(i_group_test+1)*self.infer_num_per_group]

            state, _ = self.env.reset(dir_gen, vel_gen)  # 得到环境的反馈，现在的状态
            timestep = 0
            while True:
                action = hppo_controller(state)
                action_c.extend(hppo_controller.action_c)
                period_c.extend(hppo_controller.period_c)
                action_record.extend(action)
                e_record.extend(state[:,1])
                for iiii in range(len(hppo_controller.period_c)):
                    Dpsi.append(
                        self.env.PSI[iiii, self.env.ind + hppo_controller.period_c[iiii]] - self.env.PSI[iiii, self.env.ind])
                state, reward, done, info, _ = self.env.step(action,self.wa)
                total_reward += reward
                timestep += 1

                if done:
                    break

        return action_record,e_record,action_c,period_c,Dpsi



    def control_traj_for_visualization(self, controllers, traj_num):
        """
        Generate control trajectories for visualization purposes

        Args:
            controllers: Dictionary of controllers to visualize
            traj_num: Number of trajectories to generate

        Returns:
            records: State and reward history for visualization
        """

        test_dir_gen, test_vel_gen = self.loadwind('./data/' + self.test_datafile)
        dir_gen = test_dir_gen[0:traj_num]
        vel_gen = test_vel_gen[0:traj_num]
        records = dict()
        for ind_cc in range(len(controllers.keys())):
            controller_name = list(controllers.keys())[ind_cc]
            controller = controllers[controller_name]
            state, _ = self.env.reset(dir_gen, vel_gen)  # 得到环境的反馈，现在的状态
            records[controller_name] =[[],[]]
            while True:
                action = controller(state)
                state, reward, done, info, _ = self.env.step(action, self.wa)
                records[controller_name][0].append(state)
                records[controller_name][1].append(info[0])
                if done:
                    break
        return  records