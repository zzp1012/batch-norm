# Batch Norm

## Abstract

**Overall Objective**: Rethinking batch normalization.

## Clone the project

Due to submodules, you should recursively clone the project.
```bash
    git clone --recurse-submodules -j8 git@github.com:zzp1012/batch-norm.git
```

## Requirements

1. Make sure GPU is avaible and `CUDA>=11.0` has been installed on your computer. You can check it with
    ```bash
        nvidia-smi
    ```
2. Simply create an virtural environment with `python>=3.8` and run `pip install -r requirements.txt` to download the required packages. If you use `anaconda3` or `miniconda`, you can run following instructions to download the required packages in python. 
    ```bash
        conda create -y -n BN python=3.8
        conda activate BN
        pip install pip --upgrade
        pip install -r requirements.txt
        conda activate BN
        conda install pytorch=1.10.2 torchvision=0.11.3 torchaudio=0.10.2 cudatoolkit=11.1 -c pytorch -c nvidia
    ```

## Contributing

if you would like to contribute some codes, please submit a PR, and feel free to post issues via Github.

## Contact

Please contact [zzp1012@sjtu.edu.cn](mailto:zzp1012@sjtu.edu.cn) if you have any question on the codes.
    
---------------------------------------------------------------------------------
Shanghai Jiao Tong University - Email@[zqs1022@sjtu.edu.cn](mailto:zqs1022@sjtu.edu.cn)
