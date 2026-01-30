# Wind Turbine Yaw Control with HPPO (Hierarchical Proximal Policy Optimization)

This project implements a hierarchical proximal policy optimization (HPPO) algorithm for wind turbine yaw control, including data generation, model training, inference, and cross-domain validation capabilities.

## Features

- **Data Generation**: Generate synthetic wind datasets for training and testing
- **Model Training**: Train HPPO models for yaw control optimization
- **Model Inference**: Evaluate trained models against other control strategies
- **Cross-Domain Validation**: Validate model performance across different environmental conditions
- **Stability Analysis**: Analyze stability metrics during training process

## My Environment (in fact, there is no special requirement)

- Python 3.12.7
- PyTorch 2.7.1+cu128
- NumPy 1.26.4
- Matplotlib 3.9.2

## Getting Started

### 1. Dataset Generation

#### Standard Dataset Generation
Use the `gendataset.py` script to create training and testing datasets:

```bash
python gendataset.py --num_sample 12000 --datasetname trainset
```


**Parameters:**
- `--num_sample`: Number of wind sequences to generate (default: 12000)
- `--tau_min`, `--tau_max`: Min/max tau parameter for wind generation
- `--A_min`, `--A_max`: Min/max amplitude parameter for wind generation
- `--v_cut_int`: Cut-in wind speed (m/s)
- `--v_cut_out`: Cut-out wind speed (m/s)
- `--turb_int_min`, `--turb_int_max`: Turbulence intensity range (%)
- `--roughness_min`, `--roughness_max`: Surface roughness range (m)
- `--sequence_length`: Length of each wind sequence
- `--datasetname`: Name of the generated dataset

The generated dataset will be saved in the `./data/` folder as an `.npz` file containing wind directions (`d`) and wind speeds (`v`).

#### Cross-Domain Dataset Generation
For cross-domain validation, use the `gendataset_crossdomain.py` script:

```bash
python gendataset_crossdomain.py --num_sample 12000 --datasetname testset_domain_B
```


This generates 4 domain-specific datasets with different environmental parameters, saving them as separate files in the `./data/` folder.

### 2. Configuration

Modify the configuration file `config/FCN_VEL.yml` to customize your training settings:

```yaml
WINDTURBINE: './config/wtconfig_occupy.yml'
OUTPATH: 'E:/WorkspaceY9000P2025/YawControl/HPPOResultesCrossDomain'

Net:
  StateDim: 8
  HiddenLayer: 1
  HiddenSize: 64
  ActionDim: 3
  Pretrain: 'bestmodel_domain4.pth'  # Path to pre-trained model or 'none'

Command: 'VEL'  # Options: 'VEL' (velocity) or 'ACC' (acceleration)

PPO:
  LR: 0.00001
  BETAS: [0.9, 0.999]
  GAMMA: 0.99
  K_EPOCHS: 4
  EPS_CLIP: 0.2

TRAINDATA:
  FILE_NAME: 'trainset_domain1.npz'
  GROUP_PER_EP: 1
  NUM_PER_GROUP: 8000

TESTDATA:
  FILE_NAME: 'testset_domain3.npz'
  GROUP_PER_EP: 1
  NUM_PER_GROUP: 4000

TRAIN:
  MAX_EPOCH: 100
  # ... additional training parameters
```


Key configurations you can modify:
- `OUTPATH`: Output directory for results and models
- `Net.Pretrain`: Path to pre-trained model for transfer learning or 'none'
- `TRAINDATA.FILE_NAME`: Training dataset filename
- `TESTDATA.FILE_NAME`: Testing dataset filename

### 3. Model Training

Run the training process using:

```bash
python Train.py
```


This script will:
- Load the configuration from `config/FCN_VEL.yml`
- Create output directories for logs, images, and models
- Initialize and train the HPPO controller
- Save checkpoints and results to the specified output path

### 4. Model Inference

Evaluate trained models with:

```bash
python Infer.py
```


This will:
- Load the trained model specified in the config
- Compare HPPO controller performance with other control strategies
- Generate evaluation metrics including rewards, errors, and other performance indicators
- Save inference results to the output directory

Available controllers for comparison:
- [HPPOController](file://E:\WorkspaceY9000P2025\YawControl\WTYCcodes\HPPO\WindTurbine.py#L22-L40): Hierarchical Proximal Policy Optimization controller
- [ClassicController](file://E:\WorkspaceY9000P2025\YawControl\WTYCcodes\HPPO\ClassicController.py#L0-L316): Condition-based controller
- [BangBangController](file://E:\WorkspaceY9000P2025\YawControl\WTYCcodes\HPPO2026\Models\WindTurbine.py#L9-L28): Bang-bang controller
- [OLCController](file://E:\WorkspaceY9000P2025\YawControl\WTYCcodes\HPPO\WindTurbine.py#L44-L62): Open-loop controller

### 5. Stability Analysis

Analyze the stability of all trained models during the training process:

```bash
python StableAnalysis.py
```


This script calculates stability metrics for all models generated during training, providing insights into model convergence and robustness.

## Project Structure

```
├── config/
│   ├── FCN_VEL.yml          # Main configuration file
│   └── wtconfig_occupy.yml  # Wind turbine configuration
├── data/                    # Generated datasets
├── utils/
│   └── GenWind.py           # Wind generation utilities
├── gendataset.py            # Standard dataset generator
├── gendataset_crossdomain.py # Cross-domain dataset generator
├── Train.py                 # Training script
├── Infer.py                 # Inference script
├── StableAnalysis.py        # Stability analysis script
└── Readme.md               # This documentation
```


## Usage Examples

### Generate Training Dataset
```bash
python gendataset.py --num_sample 8000 --datasetname trainset --turb_int_min 10 --turb_int_max 14
```


### Generate Test Dataset
```bash
python gendataset.py --num_sample 4000 --datasetname testset --turb_int_min 14 --turb_int_max 20
```


### Cross-Domain Dataset Generation
```bash
# For domain A (lower turbulence, lower roughness)
python gendataset_crossdomain.py --num_sample 8000 --turb_int_min 10 --turb_int_max 14 --roughness_min 100 --roughness_max 200 --datasetname trainset_domain_A

# For domain B (higher turbulence, higher roughness)
python gendataset_crossdomain.py --num_sample 4000 --turb_int_min 14 --turb_int_max 20 --roughness_min 200 --roughness_max 600 --datasetname testset_domain_B
```


### Start Training
```bash
python Train.py
```


### Run Inference
```bash
python Infer.py
```


### Perform Stability Analysis
```bash
python StableAnalysis.py
```


## Output Files

After running the scripts, the following outputs are generated:
- Trained models saved in `models/` subdirectory
- Training/inference logs in `.log` files
- Visualization plots in `imgs/` and `inferout_imgs/` subdirectories
- Performance metrics and evaluation results

## Notes

- Ensure that the `data/` directory exists before running dataset generation scripts
- The random seed is fixed for reproducible results
- Cross-domain validation requires generating multiple domain-specific datasets
- GPU support is available if CUDA-compatible hardware is present