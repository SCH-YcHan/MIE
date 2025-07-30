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
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install jupyter notebook
```
```
# git clone this repository
git clone https://github.com/SCH-YcHan/Glomer.git
cd MIE
pip install -r requirements.txt
```
```
# git clone mmsegmentation
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
pip install -U mmengine
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```
```
# Numpy version modify
pip uninstall numpy -y
pip cache purge
pip install numpy==1.24.4
```
