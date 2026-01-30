import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from HPPO2026.Models.Trainer import WTENV
from HPPO2026.Models.WindTurbine import HPPOController
import os
import datetime
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from matplotlib import rc
from matplotlib import rcParams
import matplotlib.colors as matcolors
import yaml

rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath}
\usepackage{newtxtext}    
\usepackage{mathptmx}  
'''
plt.rc('text',usetex=True)


config = {
    "font.family":'Times New Roman',
    "axes.unicode_minus": False,
    "font.size": 18.0,
    # "font.style":'italic'
}
rcParams.update(config)

def main():
    '''
    analyze the stability of all saved model of a training
    '''

    # set the path with trianing models
    models_path = r'E:\WorkspaceY9000P2025\YawControl\HPPOResults2026\Train_20260122_11_12_55_FCN_VEL\models'
    model_names = os.listdir(models_path)

    #laod the config
    cfgfile = 'FCN_VEL.yml'
    f = open(rf'./config/{cfgfile}', 'r', encoding='utf8')
    config = yaml.safe_load(f)
    f.close()

    # set the output path
    now = datetime.datetime.now()
    outpath = config[
                  'OUTPATH'] + f'\\Infer_{now.year}{now.month:02d}{now.day:02d}_{now.hour:02d}_{now.minute:02d}_{now.second:02d}_' + \
              cfgfile.split('.')[0]
    print(f'All data would be saved in {outpath}')
    os.mkdir(outpath)

    #define the print and log function
    log_file = os.path.join(outpath, config["TESTDATA"]["FILE_NAME"] +'_analysis_log.txt')
    res_file = os.path.join(outpath, config["TESTDATA"]["FILE_NAME"] + '_stable_log.txt')

    def log_print(msg):
        print(msg)
        with open(log_file,'a',encoding='utf8') as f:
            f.write(msg+'\n')
    def log_result(msg):
        print(msg)
        with open(res_file,'a',encoding='utf8') as f:
            f.write(msg+'\n')

    #analyze the running data of all model
    for model_name in model_names:
        #running the model
        config["Net"]["Pretrian"] = os.path.join(models_path, model_name)
        log_print(f'Infering model:'+config["Net"]["Pretrian"])
        wtenv = WTENV(config,outpath)
        hppocontroller = HPPOController(wtenv.infer_num_per_group, wtenv.model)
        action_record,e_record,action_c,period_c,Dpsi = wtenv.infer_hppocontroller_with_record(hppocontroller)
        action_c = [a-1 for a in action_c]
        e_data = np.array(e_record)  # 例如长度为 1000
        d_data = np.array(action_c)  # 动作序列，值为 -1, 0, 1
        h_data = np.array(period_c)
        Dpsi_data = np.array(Dpsi)
        np.savez(f'{outpath}/my_data.npz', e_data=e_data, d_data=d_data, h_data=h_data, Dpsi_data=Dpsi_data)

        C1 = 0.7 / 57.3 #the max pitch velocity

        # intermediate variables
        ed = e_data * d_data  # e_k * d_k
        hd = h_data * d_data  # h_k * d_k
        h2 = h_data ** 2  # h_k^2
        d2 = d_data ** 2  # d_k^2

        e1__data = e_data - C1 * h_data * d_data
        e1_data = e_data - C1 * h_data * d_data + Dpsi_data
        E_dv_ = np.mean(0.5 * e1__data ** 2 - 0.5 * e_data ** 2)
        E_dv = np.mean(0.5 * e1_data ** 2 - 0.5 * e_data ** 2)
        # print('Orgninal Conditon of DV_ without dpsi: np.mean(0.5*e1__data**2-0.5*e_data**2) = {}'.format(E_dv_))
        # print('Orgninal Conditon DV with psi: np.mean(0.5*e1_data**2-0.5*e_data**2) = {}'.format(E_dv))

        log_print('Original Lyapunov function condtion：E[DV-] = {}, E[DV] ={}'.format(E_dv_, E_dv))

        # covariance matrix
        cov_he = np.cov(h_data, e_data)[0, 1]
        cov_de = np.cov(d_data, e_data)[0, 1]
        cov_h_ed = np.cov(h_data, ed)[0, 1]
        cov_dh = np.cov(d_data, h_data)[0, 1]  # Cov(d,h) = Cov(h,d)
        cov_h2_d2 = np.cov(h2, d2)[0, 1]
        cov_dpsi_hd = np.cov(Dpsi_data, hd)[0, 1]
        cov_e_dpsi = np.cov(e_data, Dpsi_data)[0, 1]

        #all mean
        E_e = np.mean(e_data)  # E[e_k]
        E_e2 = np.mean(e_data ** 2)  # E[e_k^2]
        E_hd = np.mean(hd)  # E[h_k * d_k]
        E_ed = np.mean(ed)  # E[e_k * d_k]
        E_Dpsi2 = np.mean(Dpsi_data ** 2)
        E_Dpsi = np.mean(Dpsi_data)

        E_h = np.mean(h_data)
        E_h2 = np.mean(h_data ** 2)
        E_d = np.mean(d_data)
        E_d2 = np.mean(d_data ** 2)

        E_eDpsi = np.mean(e_data * Dpsi_data)
        E_ehd = np.mean(e_data * hd)
        E_h2d2 = np.mean(h2 * d2)
        E_Dpsihd = np.mean(Dpsi_data * hd)

        # standard deviation
        sigma_h = np.std(h_data, ddof=1)  # std(h)
        sigma_e = np.std(e_data, ddof=1)  # std(e)
        sigma_ed = np.std(ed, ddof=1)  # std(e*d)
        sigma_d = np.std(d_data, ddof=1)  # std(d)
        sigma_d2 = np.std(d2, ddof=1)  # std(d)
        sigma_h2 = np.std(h2, ddof=1)  # std(d)
        sigma_dpsi = np.std(Dpsi_data, ddof=1)  # std(Dpsi)
        sigma_hd = np.std(hd, ddof=1)

        # correlation
        rho_he = cov_he / (sigma_h * sigma_e) if sigma_h > 1e-8 and sigma_e > 1e-8 else 0.0
        rho_h_ed = cov_h_ed / (sigma_h * sigma_ed) if sigma_h > 1e-8 and sigma_ed > 1e-8 else 0.0
        rho_de = cov_de / (sigma_d * sigma_e) if sigma_d > 1e-8 and sigma_e > 1e-8 else 0.0
        rho_dh = cov_dh / (sigma_d * sigma_h) if sigma_d > 1e-8 and sigma_h > 1e-8 else 0.0
        rho_h2_d2 = cov_h2_d2 / (sigma_d2 * sigma_h2) if sigma_d2 > 1e-8 and sigma_h2 > 1e-8 else 0.0
        rho_e_dpsi = cov_e_dpsi / (sigma_e * sigma_dpsi) if sigma_e > 1e-8 and sigma_dpsi > 1e-8 else 0.0
        rho_dpsi_hd = cov_dpsi_hd / (sigma_dpsi * sigma_hd) if sigma_dpsi > 1e-8 and sigma_hd > 1e-8 else 0.0


        log_print("\n" + "-" * 20)
        log_print("Cov")
        log_print("=" * 50)
        log_print(f"Cov(h, e)         = {cov_he: .8f}")
        log_print(f"Cov(h, e*d)       = {cov_h_ed: .8f}")
        log_print(f"Cov(d, e)         = {cov_de: .8f}")
        log_print(f"Cov(d, h)         = {cov_dh: .8f}")
        log_print(f"Cov(h^2, d^2)     = {cov_h2_d2: .8f}")
        log_print(f"Cov(Dpsi, h*d)      = {cov_dpsi_hd: .8f}")
        log_print(f"Cov(e, Dpsi)        = {cov_e_dpsi: .8f}")

        # standard deviation
        sigma_h = np.std(h_data, ddof=1)  # std(h)
        sigma_e = np.std(e_data, ddof=1)  # std(e)
        sigma_ed = np.std(ed, ddof=1)  # std(e*d)
        sigma_d = np.std(d_data, ddof=1)  # std(d)
        sigma_dpsi = np.std(Dpsi_data, ddof=1)  # std(Dpsi)
        sigma_hd = np.std(hd, ddof=1)
        log_print("\n" + "-" * 20)
        log_print(f"sigma_h              = {sigma_h: .8f}")
        log_print(f"sigma_e              = {sigma_e: .8f}")
        log_print(f"sigma_ed              = {sigma_ed: .8f}")
        log_print(f"sigma_d              = {sigma_d: .8f}")
        log_print(f"sigma_Dpsi              = {sigma_dpsi: .8f}")
        log_print(f"sigma_hd              = {sigma_hd: .8f}")

        log_print("\n" + "-" * 20)
        log_print("rho")
        log_print("=" * 50)
        log_print(f"rho(h, e)         = {rho_he:.4f}")
        log_print(f"rho(h, e*d)       = {rho_h_ed:.4f}")
        log_print(f"rho(d, e)         = {rho_de:.4f}")
        log_print(f"rho(d, h)         = {rho_dh:.4f}")
        log_print(f"rho(d2, h2)         = {rho_h2_d2:.4f}")
        log_print(f"rho(Dpsi, h*d)      = {rho_dpsi_hd: .8f}")
        log_print(f"rho(e, Dpsi)        = {rho_e_dpsi: .8f}")
        log_print("\n" + "-" * 20)
        log_print(f"E[e]              = {E_e: .8f}")
        log_print(f"E[e^2]            = {E_e2: .8f}")
        log_print(f"E[h*d]            = {E_hd: .8f}")
        log_print(f"E[e*d]            = {E_ed: .8f}")
        log_print(f"E[h]            = {E_h: .8f}")
        log_print(f"E[h2]            = {E_h2: .8f}")
        log_print(f"E[d]            = {E_d: .8f}")
        log_print(f"E[d2]            = {E_d2: .8f}")
        log_print(f"E[Dpsi]            = {E_Dpsi: .8f}")
        log_print(f"E[Dpsi2]            = {E_Dpsi2: .8f}")

        N = len(e_data)
        assert len(d_data) == N and len(h_data) == N, "数据长度不一致！"
        assert np.all(np.isin(d_data, [-1, 0, 1])), "d_data 应只包含 -1, 0, 1"
        log_print("\n" + "-" * 20)
        log_print(f"Total number of samples: {N}")
        log_print(f"Action distribution: d=-1: {np.sum(d_data == -1)}, d=0: {np.sum(d_data == 0)}, d=1: {np.sum(d_data == 1)}")

        e_given_d1 = e_data[d_data == 1]
        e_given_dm1 = e_data[d_data == -1]
        e_given_d0 = e_data[d_data == 0]


        # original condition
        BasicConditon = E_eDpsi - C1 * E_ehd + 0.5 * C1 ** 2 * E_h2d2 - C1 * E_Dpsihd
        log_print('BasicConditon = {}'.format(BasicConditon))
        str_res = 'system is stable with BasicConditon' if BasicConditon < 0 else 'system is unstable with BasicConditon'
        log_print(str_res)

        # Condition with Assumption 1
        AssumptionConditon = - C1 * E_ehd + 0.5 * C1 ** 2 * E_h2d2
        log_print('AssumptionConditon = {}'.format(AssumptionConditon))
        str_res = 'system is stable with AssumptionConditon' if AssumptionConditon < 0 else 'system is unstable with AssumptionConditon'
        log_print(str_res)

        res_str = model_name
        res_str += '| base condition: ' + str(BasicConditon)
        res_str += ',   stable' if BasicConditon < 0 else ', unstable'
        res_str += '|  ' + str(AssumptionConditon)
        res_str += ',   stable' if AssumptionConditon < 0 else ' unstable'
        log_result(res_str)

        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = default_colors[:4]  # ['#1f77b4', '#ff7f0e', '#2ca02c', ...]

        #plot conditional density of yaw direction command
        plt.figure(figsize=(6.5, 5), dpi=600)
        mu, std = stats.norm.fit(e_data)
        x = np.linspace(-1, 1, 1000)
        p = stats.norm.pdf(x, mu, std)
        num_bins = 50
        bins = np.linspace(-1, 1, num_bins + 1)  # 注意：N 个区间需要 N+1 个边界点
        plt.hist(e_given_d1, bins=bins, density=False, alpha=0.6, color=colors[0],
                 label=r'$\mathrm{P}(e_\kappa|d_\kappa=1)$')
        plt.hist(e_given_d0, bins=bins, density=False, alpha=0.6, color=colors[1],
                 label=r'$\mathrm{P}(e_\kappa|d_\kappa=0)$')
        plt.hist(e_given_dm1, bins=bins, density=False, alpha=0.6, color=colors[2],
                 label=r'$\mathrm{P}(e_\kappa|d_\kappa=-1)$')

        plt.xlabel(r'$e_\kappa$ (rad)', fontsize=18)
        plt.ylabel('Count', fontsize=18)

        plt.legend()
        plt.xlim([-0.75, 0.75])
        # plt.axis([-1, 1, 0, 10])
        plt.tight_layout()
        plt.savefig(f'{outpath}/conditiondensity.png')
        plt.close()
        plt.figure(figsize=(6.5, 5), dpi=600)

        #plot conditional density of yaw period command
        h_values = np.arange(7, 15)
        colors = plt.cm.viridis(np.linspace(0, 1, len(h_values)))
        num_bins = 50
        bins = np.linspace(-1, 1, num_bins + 1)  # 范围 [-1, 1] 弧度
        log_print("\n" + "-" * 20)
        for i, h in enumerate(h_values):
            e_given_h = e_data[h_data == h]

            if len(e_given_h) == 0:
                log_print(f"No data for h_kappa = {h}")
                continue
            plt.hist(e_given_h, bins=bins, density=True, alpha=0.7, color=colors[i],
                     label=fr'$\mathrm{{P}}(e_\kappa \mid h_\kappa={h})$')

        plt.xlabel(r'$e_\kappa$ (rad)', fontsize=18)
        plt.ylabel('Density', fontsize=18)
        plt.xlim([-0.75, 0.75])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{outpath}/condition_density_by_h.png')
        plt.close()


        # plot 3d view of tracking error, yaw direction command, and yaw period command
        fig = plt.figure(figsize=(10, 8), dpi=600)
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(e_data[::1000], d_data[::1000], h_data[::1000], c=h_data[::1000], cmap='viridis',
                        s=50)  #
        ax.view_init(elev=10, azim=-56)
        ax.set_zlabel(r'$h_\kappa (s)$', rotation=-90, fontsize=24, labelpad=10)

        ax.set_xlabel(r'$e_\kappa (rad)$', fontsize=24, labelpad=20)
        ax.set_ylabel(r'$d_\kappa$', fontsize=24, labelpad=10)

        norm = matcolors.Normalize(vmin=7, vmax=15)
        ax.set_xlim([-0.75, 0.75])
        ax.set_ylim([-1, 1])
        ax.set_zlim([6, 16])
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['-1', '0', '1'], fontsize=24)  # 设置 y 轴标签及字号
        ax.set_xticks([-0.6, -0.3, 0.0, 0.3, 0.6])
        ax.set_xticklabels(['-0.6', '-0.3', '0.0', '0.3', '0.6'], fontsize=24)  # 设置 x 轴标签及字号
        ax.set_zticks([6, 8, 10, 12, 14, 16])
        ax.set_zticklabels(['6', '8', '10', '12', '14', '16'], fontsize=24)  # 设置 x 轴标签及字号
        colorbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=20, pad=0.02, location='left',
                                norm=norm)  # 使用 shrink 参数调整颜色条大小
        colorbar.set_ticks([7, 9, 11, 13, 15])  # 设置颜色条的刻度
        colorbar.set_ticklabels(['7', '9', '11', '13', '15'])  # 设置颜色条的标签
        colorbar.ax.tick_params(labelsize=24)
        plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
        plt.savefig(f'{outpath}/e_d_h.png')
        plt.close()
        log_print("\n" + "=" * 50)

if __name__ == '__main__':
    main()


