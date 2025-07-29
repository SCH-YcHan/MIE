# MIE
MIE: Magnification Integrated Ensemble method for improving glomeruli segmentation in medical imaging

## Env setting (Local)
```
OS: Windows 11 Pro
CPU: 12th Gen Intel(R) Core(TM) i7-12700F 
GPU: NVIDIA GeForce RTX 4070
CUDA version: 11.8
CuDNN version: 8.4.0
Workstation: Anaconda3
```
```
# Anaconda env setting
conda create -n MIE python=3.10
activate MIE
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install jupyter notebook
```
```
# git clone this repository
git clone https://github.com/SCH-YcHan/Glomer.git
cd MIE
```
```
# git clone mmsegmentation
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```
