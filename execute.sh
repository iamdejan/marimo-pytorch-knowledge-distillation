#!/bin/bash
#SBATCH --partition=gpu-v100s
#SBATCH --job-name=marimo-pytorch-knowledge-distillation
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --qos=long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=24201613@siswa.um.edu.my
#SBATCH --gpus=1

# install Pixi
curl -fsSL https://pixi.sh/install.sh | sh

# install dependencies
pixi install

# execute notebook
pixi run execute practice.py
