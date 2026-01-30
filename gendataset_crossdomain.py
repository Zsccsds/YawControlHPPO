import matplotlib.pyplot as plt
import argparse
from utils.GenWind import *

def main():
    parser = argparse.ArgumentParser(description="Generate wind dataset for yaw control training/testing.")
    parser.add_argument('--num_sample', type=int, default=12000,
                        help="Number of wind sequences to generate")

    parser.add_argument('--tau_min', type=int, default=200,
                        help="min tau for GenWind function")
    parser.add_argument('--tau_max', type=int, default=1600,
                        help="max tau for GenWind function")
    parser.add_argument('--A_min', type=int, default=1,
                        help="max A for GenWind function")
    parser.add_argument('--A_max', type=int, default=40,
                        help="max A for GenWind function")

    parser.add_argument('--v_cut_int', type=int, default=3,
                        help="Cut-in wind speed, m/s")
    parser.add_argument('--v_cut_out', type=int, default=25,
                        help="Cut-out wind speed, m/s")
    parser.add_argument('--turb_int_min', type=int, default=10,
                        help="Turbulence intensity, %")
    parser.add_argument('--turb_int_max', type=int, default=20,
                        help="Turbulence intensity, %")
    parser.add_argument('--roughness_min', type=int, default=100,
                        help="Roughness, m")
    parser.add_argument('--roughness_max', type=int, default=500,
                        help="Roughness, m")
    parser.add_argument('--sequence_length', type=int, default=1400,
                        help="Length of each wind sequence")

    parser.add_argument('--datasetname', type=str, default='testset',
                        help="the name of generated dateset ")


    args = parser.parse_args()
    n = int(args.num_sample)

    print(f'Generate {n} samples for each condition for {args.datasetname}')
    wind_dirs,wind_vs =[[],[],[],[]],[[],[],[],[]] # 4 domains

    # generate n*4 samples
    for _ in range(n):
        Tau = np.random.randint(args.tau_min,args.tau_max)
        A = np.random.randint(args.A_min,args.A_max)
        v_mean = np.random.randint(args.v_cut_int, args.v_cut_out)
        turb_int = np.random.randint(args.turb_int_min, args.turb_int_max)
        roughness = np.random.randint(args.roughness_min, args.roughness_max)
        print(f'Generating wind dataset for Tau={Tau}, A={A}, v_mean={v_mean},turb_int={turb_int},roughness={roughness}')
        d, v = genwind(1, Tau, A, v_mean,turb_int)
        #divide 4 domains
        for i in range(4):
            wind_dirs[i].append(d[i] + np.random.uniform(-90,90))
            wind_vs[i].append(v[i])

    print(f'Generated {len(wind_dirs[0])} samples in total.')
    for ii in range(4):
        dirs = np.array(wind_dirs[ii])
        vs = np.array(wind_vs[ii])
        plt.figure(figsize=(24,15))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        # plot 12 wind direction profiles
        for j in range(12):
            plt.subplot(3,4,j+1)
            plt.plot(dirs[j,:])
        plt.savefig(f'./data/d_domain{ii+1}.png',dpi=600)

        plt.figure(figsize=(24,15))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        # plot 12 wind speed profiles
        for j in range(12):
            plt.subplot(3,4,j+1)
            plt.plot(vs[j,:])
        plt.savefig(f'./data/v_domain{ii+1}.png',dpi=600)
        plt.close()

        np.savez(f'./data/trainset_domain{ii+1}.npz',d=np.array(wind_dirs[ii][0:8000]),v=np.array(wind_vs[ii][0:8000]))
        np.savez(f'./data/testset_domain{ii+1}.npz',d=np.array(wind_dirs[ii][8000:]),v=np.array(wind_vs[ii][8000:]))
if __name__=='__main__':
    main()
