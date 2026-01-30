import matplotlib.pyplot as plt
import argparse
from utils.GenWind import *


def main():
    """
    Main function to generate wind dataset for yaw control training and testing
    - Parses command line arguments for dataset generation parameters
    - Generates wind direction and velocity data with random variations
    - Saves the generated data and creates preview plots
    """

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

    parser.add_argument('--datasetname', type=str, default='trainset',
                        help="the name of generated dateset ")


    args = parser.parse_args()
    n = int(args.num_sample/4) # Calculate number of samples needed (divided by 4 since genwind_noise generates 4 sequences per call)
    print(f'Generate {n} samples for each condition for {args.datasetname}')
    wind_dirs,wind_vs =[],[]

    # Generate wind data with random parameters within specified ranges
    for _ in range(n):
        Tau = np.random.randint(args.tau_min,args.tau_max)
        A = np.random.randint(args.A_min,args.A_max)
        v_mean = np.random.randint(args.v_cut_int, args.v_cut_out)
        turb_int = np.random.randint(args.turb_int_min, args.turb_int_max)
        roughness = np.random.randint(args.roughness_min, args.roughness_max)
        print(f'Generating wind dataset for Tau={Tau}, A={A}, v_mean={v_mean},turb_int={turb_int},roughness={roughness}')
        # Generate wind data with nois
        d, v = genwind_noise(1, Tau, A, v_mean,turb_int)
        wind_dirs.extend(d)
        wind_vs.extend(v)
    print(f'Generated {len(wind_dirs)} samples in total.')

    # Create preview plots for first 10 samples (12 subplots showing every 40th sample)
    dirs = np.array(wind_dirs[0:10:1])
    vs = np.array(wind_vs[0:10:1])
    plt.figure(figsize=(24,15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Plot wind directions
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.plot(dirs[i*40,:])
    plt.savefig(f'./data/{args.datasetname}_d.png',dpi=600)

    plt.figure(figsize=(24,15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Plot wind velocities
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.plot(vs[i*40,:])
    plt.savefig(f'./data/{args.datasetname}_v.png',dpi=600)
    plt.close()

    # Convert lists to numpy arrays and save as compressed file
    wind_dirs = np.array(wind_dirs)
    wind_vs = np.array(wind_vs)
    np.savez(f'./data/{args.datasetname}.npz',d=wind_dirs,v=wind_vs)

if __name__=='__main__':
    main()
