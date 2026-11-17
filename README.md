# 2.5D Transformer: An Efficient 3D Seismic Interpolation Method without Full 3D Training

## Introduction

This repository contains the implementation of "2.5D Transformer: An Efficient 3D Seismic Interpolation Method 
without Full 3D Training".

- Paper: [arXiv:2511.10033](https://doi.org/10.48550/arXiv.2511.10033)
- Supplementary details: [supplementary_details.md](./supplementary_details.md)

The method consists of two training stages. In Stage 1, a Transformer-based network (named T-2D) is trained on 
2D patches. In Stage 2, its Transformer encoder parameters are transferred to a 3D network (named T-2.5D) 
and frozen, while the seismic dimension adapters (SDAs) and the remaining modules are trained on 3D volumes.

**Attention:** Only the Kerry dataset are provided here as an example. You may visit http://wiki.seg.org 
for more open datasets that appear in our paper, or you can get a copy of processed '.mat' files by
contacting cxwei24@mails.jlu.edu.cn.

All the provided seismic datasets are saved in '.mat' format for simplicity. If you intend to use your 
own datasets, please make sure the filename is consistent with the variable name inside the '.mat' file.

All training data should be saved in `'./data/[name_for_your_data]/train/'`. All test data should be saved 
in `'./data/[name_for_your_data]/test/'`.

---

## File Introduction

```text
Codes/
├── README.md                              # READ THIS FIRST!!!
├── requirements.txt                       # required packages  
├── supplementary_details.md               # supplementary experimental details
├── __init__.py
│
├── src/
│   ├── train.py                            # 2D pretraining and full 3D training
│   ├── test_2d.py                          # testing program for 2D interpolation
│   ├── test_3d.py                          # testing program for 3D interpolation
│   ├── utils.py                           # utilities
│   ├── logger.py                          # logging utilities
│   │
│   ├── dataset_processing/
│   │   ├── dp2d.py                     # 2D patches generation
│   │   ├── dp3d.py                     # 3D volumes generation
│   │   └── dcls.py                     # some classes for loading datasets
│   │
│   ├── transfer_learning/
│   │   ├── tl_2d_3d.py                 # Stage 2 transfer learning
│   │   ├── fine_tuning.py              # target-survey full fine-tuning
│   │   └── params_load.py                 # parameter loading and freezing
│   │
│   ├── networks/
│   │   ├── 2d/T2D.py                   # 2D Transformer
│   │   ├── 3d/T3D.py                   # full 3D Transformer
│   │   └── tl/                         # transfer learning networks and adapters
│   │       ├── T25D.py
│   │       ├── Seismic_3D_Adapter.py
│   │       └── lib/PatchEmbedUnembed_t.py 
│   │
│   └── conventional_methods/fk_POCS_3d.py # conventional f-k POCS interpolation
│   
├── shell_scripts/                      # see instructions below for usage
│   ├── dataset_processing.sh        
│   ├── train_T2D.sh                
│   ├── train_T3D.sh           
│   ├── train_T2.5D.sh             
│   ├── test_2d.sh           
│   ├── test_3d.sh           
│   ├── test_2.5d.sh              
│   ├── full_finetuning_T2.5D.sh               
│   ├── full_finetuning_T3D.sh             
│   └── run_pocs.sh                        
│  
├── data/                           # .mat data, not provided
│   ├── kerry/                      # Kerry dataset
│   │   ├── train                   # data for training
│   │   │   └── kerry3d_train.mat           
│   │   └── test                    # data for testing
│   │       └── kerry3d_test.mat    
│   │
│   ├── parihaka/                   # the same file organization with Kerry
│   │   ├── ...
│   │   └── ...
│   │
│   ├── opunake/                   # the same file organization with Kerry
│   │   ├── ...
│   │   └── ...
│   │
│   └── overthrust/                # the same file organization with Kerry
│       ├── ...
│       └── ...
├── dataset/                        # prepared dataset used for training
│   ├── kerry_2D_40_6k.h5   
│   │   ├── train                   # architecture within the ".h5" file
│   │   │   ├── label
│   │   │   │   ├── patch1
│   │   │   │   ├── patch2
│   │   │   │   ├── ...
│   │   │   │   └── patchN
│   │   │   │   
│   │   │   └── mask
│   │   │       ├── mask1          # correspond to the patches in train/label
│   │   │       ├── mask2
│   │   │       ├── ...
│   │   │       └── maskN
│   │   │
│   │   └── val
│   │       └── ...                 # the same architecture with train     
│   └── ...   
│ 
├── trained_models/
│   ├── 2d/                                # Stage 1/T-2D models
│   ├── 3d/                                # full 3D models
│   ├── tl/                                # Stage 2 models and full fine-tuning models
│   └── pretrained/                        # models copied here for 2D-to-3D TL
│ 
├── logs/                                  # training logs
├── results/                               # interpolation results and metrics 
└── matlab_scripts/                 # matlab scripts for figure generation
    ├── display_3d_kerry.m        # display a 3D view of SEG C3 45shots dataset 
    ├── display_2d_kerry.m        # display a 2D view of SEG C3 45shots dataset 
    ├── display_fk_kerry.m        # display f-k spectrum of SEG C3 45shots dataset 
    ├── drr_plot3d.m                # 3D figure function
    ├── fk.m                        # f-k figure function 
    ├── cseis.m                     # color map function for seismic data 
    └── blue_white_red.mat          # color map for seismic data 
```

---

## How to run training and testing

Please follow these instructions step-by-step to train and test T-2.5D successfully.

### 1. Environment

#### 1.1 Installment of Anaconda and Python 
It is recommended to install Anaconda, which comes with Python pre-installed.

**Linux-OS** is recommended (not recommended on Windows).

① Here is an example for installing Python using Anaconda.
```shell
# 1. Download Anaconda in any directory (Specific Version
# can be found in the official website: https://www.anaconda.com/)
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh

# 2. Setup Anaconda by
bash ./Anaconda3-2024.06-1-Linux-x86_64.sh

# 3. Add Anaconda3's bin path to environment variable
# (You should replace the path in the following command)
echo 'export PATH="/path/to/your/anaconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 4. Test your installation of Anaconda.
conda --version

# 5. Python3 and Pip3 are integrated in Anaconda, test them.
python --version
pip --version
```
It is recommended to create a new environment as below
(not recommended on the base environment).

② For the first time you use Anaconda, you should:
```shell
conda init
```
otherwise, skip to ③.

③ Reboot your terminal.

④ Create an environment named 'testenv' (or any name you like)
, with python 3.12.3:
```shell
conda create -n testenv python=3.12.3
```
⑤ Activate 'testenv':
```shell
conda activate testenv
```

#### 1.2 Install all required packages
```shell
pip install -r requirements.txt
```

---

### 2. Dataset preparation
A '.mat' file should be prepared in `./data/[dataset]/train/`, and the variable
inside the file should be consistent with the filename. 

To obtain training and validation datasets, you should run the following command
to process the original seismic data into data patches and save them as
[complete]-[incomplete] pairs into '.h5' file.

Run the shell script as follows:
``` shell
bash ./shell_scripts/dataset_processing.sh [dataset] 
```
where

`[dataset]: "kerry", "parihaka", "opunake", or "overthrust"`

After finishing this script, 2D dataset of shape 40×40 and 3D dataset of 32×32×32 
will be created at the same time, named by `[dataset]_2D_40_6k.h5` 
and `[dataset]_3D_404016_6k.h5`, respectively, and both saved to `'./dataset'`.

This output indicates a successful processing:
```shell
Training set: 6000 patches
Validation set: 1500 patches
...

torch.Size([1, 1, 40, 40])
torch.Size([1, 1, 40, 40])
...

Training set: 6000 patches
Validation set: 1500 patches
...

torch.Size([1, 1, 40, 40, 16])
torch.Size([1, 1, 40, 40, 16])

```
---

### 3. Training

After finishing Step 1 and Step 2, you may start training.

#### 3.1 Stage 1/T-2D
```shell
bash ./shell_scripts/train_T2D.sh [name of your prepared dataset]
```
where

`[name of your prepared dataset]`: modify this to the filename 
(without extension) of your prepared dataset

For example,
```shell
bash ./shell_scripts/train_T2D.sh kerry_2D_40_6k
```
The trained models will be saved to `'./trained_models/2d/'.`

#### 3.2 Stage 2
Before Stage 2 training, there should already exist a trained 2D model 
copied into `'./trained_models/pretrained/'`.

Run the following command to start Stage 2 training:
```shell
bash ./shell_scripts/train_T2.5D.sh [name of your prepared dataset] [name of your trained model]
```
where both params are without extensions.

For example,
```shell
bash ./shell_scripts/train_T2.5D.sh kerry_3D_404016_6k kerry2D
```
The trained models will be saved to `'./trained_models/tl/'`.

#### 3.3 Others
For T-3D, use the following command for training:
```shell
bash ./shell_scripts/train_T3D.sh [name of your prepared dataset]
```
The trained models will be saved to `'./trained_models/3d/'`.

For full fine-tuning, use the following command:
```shell
bash ./shell_scripts/full_finetuning_T2.5D.sh [name of your prepared dataset] [name of your trained model]
bash ./shell_scripts/full_finetuning_T3D.sh [name of your prepared dataset] [name of your trained model]
```
The trained models will be saved to `'./trained_models/tl/'`.

### 4. Testing
After finishing training, test them using:
```shell
bash ./shell_scripts/test_2D.sh [dataset/testfolder] [foldername of your trained model]
bash ./shell_scripts/test_2.5D.sh [dataset/testfolder] [foldername of your trained model]
bash ./shell_scripts/test_3D.sh [dataset/testfolder] [foldername of your trained model]
```
where

`[dataset/testfolder]`: replace `dataset` with "kerry", "parihaka", "opunake", or "overthrust";
and replace `testfolder` with the subfolder of the corresponding dataset

`[foldername of trained model]`: modify this to the foldername of your trained model

For example:
```shell
bash shell_scripts/test/test_2D.sh kerry/test_2d TPNN T2.5D_example
bash shell_scripts/test/test_2.5D.sh kerry/test_3d TPNN T2.5D_example
```

**Attention**: The 2D test data must contain at least two seismic lines, with each
line saved into an individual file. The files should be placed in the same folder,
with each filename ending in a number indicating its relative order.

The interpolated results will be saved to `'./results/2d/'` or `'./results/3d/'` in '.mat' format.
The 2D test results will be automatically combined into an entire 3D volume if correctly tested.

There will be four variables in one '.mat'-format result:

| Variable | Description |
|---|---|
| `complete` | Complete reference data |
| `missing` | Input data with missing traces |
| `reconstructed` | Interpolated data |
| `difference` | `reconstructed - complete` |

\
Additionally, the f-k POCS methods are also provided. Run it like this:
```shell
bash shell_scripts/run_pocs.sh kerry/test_3d
```

---

### 5. Figure Generation
We use Matlab R2023a to generatre figures.
Part of the matlab scripts are provided in `./matlab_scripts/`.

**The three '.m' scripts that start with `'display_'` are mainly
used to generate the figures of the Kerry dataset in our paper, corresponding to 
the 3D views, 2D views, and f-k spectra, respectively.**

To generate figures, you should first load the interpolated results
to the 'Workspace'. The four variables introduced in the last subsection
should be available in the 'Workspace'.

Each displaying script begins with the following content:
```
data = complete;

% data = missing;

% data = reconstructed;

% data = difference;
```

Uncomment one of the statements, 
comment out the others, 
and then run the entire displaying script 
to generate the corresponding figure.
