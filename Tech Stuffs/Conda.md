# Conda

## Installation
Download the installation script
```bash
cd ~

curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
```

Install conda
```bash
chmod +x Miniforge3-Linux-x86_64.sh

./Miniforge3-Linux-x86_64.sh
```

During installation, say `yes` to add conda to shell.
Then, disable `base env` initialization with
```bash
conda config --set auto_activate_base false
```

## Usage
Creating the environment.
```bash
conda create -y -n lerobot python=3.10
```

Activating the environment.
```bash
conda activate lerobot
```


