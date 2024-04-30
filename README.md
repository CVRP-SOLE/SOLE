<!-- PROJECT LOGO -->

<p align="center">
  <h1 align="center">Segment Any 3D Object with Language</h1>
  <p align="center">
    <a href="https://github.com/0nandon">Seungjun Lee</a><sup>1*</sup></span> · 
    <a href="https://yuyangzhao.com">Yuyang Zhao</a><sup>2*</sup> · 
    <a href="https://www.comp.nus.edu.sg/~leegh/">Gim Hee Lee</a><sup>2</sup> <br>
    <sup>1</sup>Korea University · 
    <sup>2</sup>National University of Singapore<br>
    <sup>*</sup>equal contribution
  </p>
  <h2 align="center">arXiv 2024</h2>
  <h3 align="center"><a href="https://github.com/CVRP-SOLE/SOLE">Code</a> | <a href="https://arxiv.org/abs/2404.02157">Paper</a> | <a href="https://cvrp-sole.github.io">Project Page</a> </h3>
  <div align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
  <a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
  </div>
</p>
<p align="center">
  <a href="">
    <img src="https://github.com/CVRP-SOLE/CVRP-SOLE.github.io/blob/main/static/images/teaser.png?raw=true" alt="Logo" width="100%">
  </a>
</p>
<p align="center">
<strong>SOLE</strong> is highly generalizable and can segment corresponding instances with various language instructions, including but not limited to visual questions, attributes description, and functional description.
</p>
<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#todo">TODO</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li>
      <a href="#weights">Weights</a>
    </li>
    <li>
      <a href="#training-and-testing">Training and Testing</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>

## News:

- [2024/04/20] Code is released.

## TODO
- [x] Release the code
- [ ] Release the preprocessed data
- [ ] Release the evaluation code for Replica dataset
- [ ] Release the preprocessed data and precomputed features for Replica dataset

## Installation

### Dependencies :memo:
The main dependencies of the project are the following:
```yaml
python: 3.10.9
cuda: 11.3
```
You can set up a conda environment as follows
```
export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"

conda env create -f environment.yml

conda activate sole

pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip3 install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

mkdir third_party
cd third_party

git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

cd ../../pointnet2
python setup.py install

cd ../../
pip3 install pytorch-lightning==1.7.2
pip3 install open-clip-torch
```

## Data Preparation

We provide the **pre-processed 3D data** and **precomputed features** for the training and evaluation. You can download each data from the corresponding link below:
- Pre-processed 3D data
  * <a href="">ScanNet</a>
  * <a href="">ScanNet200</a>
  * ~~Replica~~ (coming soon)
- Precomputed per-point CLIP features
  * <a href="">ScanNet</a>
  * ~~Replica~~ (coming soon)
- Precomputed features of MCA and MEA
  * <a href="">ScanNet</a>
  * <a href="">ScanNet200</a>

Once you download the data, locate each files to the corresponding path indicated below:
```
├── data
│   └── preprocessed
│       ├── scannet                   <- Preprocessed ScanNet data
│       └── scannet200                <- Preprocessed ScanNet200 data
│   
├── openvocab_supervision
│   ├── openseg     
│   │   └── scannet                   <- Precomputed per-point CLIP features for ScanNet
│   │       ├── scene0000_00_0.pt
│   │       ├── scene0000_01_0.pt
│   │       └── ...
│   ├── scannet_mca                   <- Precomputed features of MCA for ScanNet
│   │   ├── scene0000_00.pickle
│   │   ├── scene0000_01.pickle
│   │   └── ...
│   ├── scannet_mea                   <- Precomputed features of MEA for ScanNet
│   │   ├── scene0000_00.pickle
│   │   ├── scene0000_01.pickle
│   │   └── ...
│   ├── scannet200_mca                <- Precomputed features of MCA for ScanNet200
│   │   ├── scene0000_00.pickle
│   │   ├── scene0000_01.pickle
│   │   └── ...
│   └── scannet200_mea                <- Precomputed features of MEA for ScanNet200
│       ├── scene0000_00.pickle
│       ├── scene0000_01.pickle
│       └── ...   
```

## Weights

For the stable training, we employ a two-stage training process:

1. Pretrain the backbone with only using mask-annotations.
2. Train the mask decoder while backbone is fixed. Mask annotations and three types of associations are used for the training. (See the original paper for the details.)

For the training, we provide pretrained backbone weights for ScanNet and ScanNet200 datasets. Download the weights from the link bellow:
- <a href="">Backbone weights for ScanNet</a>
- <a href="">Backbone weights for ScanNet200</a>

Locate the downloaded backbone weights under the `backbone_checkpoint` folder like below:
```
├── backbone_checkpoint
│   ├── backbone_scannet.ckpt        <- Backbone weights for ScanNet
│   └── backbone_scannet200.ckpt     <- Backobne weights for ScanNet200
```
Once you download the provided data <a href="#data-preparation">here</a> and pretrained backbone weights, you are ready to train the model. Check the training command in <a href="#training-and-testing">Training and Testing</a> section.

For the evaluation, we provide the official weight of SOLE for ScanNet and ScanNet200 datasets. Download the weights from the link below:
- <a href="">Offical weights of SOLE for ScanNet</a>
- <a href="">Official weights of SOLE for ScanNet200</a>
- ~~Official weights of SOLE for Replica~~ (coming soon)

Once you download the weights, locate the downloaded weights under the `checkpoint` folder like below:
```
├── checkpoint
│   ├── scannet.ckpt        <- Official weights for ScanNet
│   └── scannet200.ckpt     <- Official weights for ScanNet200
```
Now you are ready to evaluate the model. Check the evaluation command in <a href="#training-and-testing">Training and Testing</a> section.

## Training and Testing
Train the SOLE on the ScanNet dataset.
```
bash scripts/scannet/scannet_train.sh
```
Train the SOLE on the ScanNet200 dataset.
```
bash scripts/scannet200/scannet200_train.sh
```
Evaluate the SOLE on the ScanNet dataset.
```
bash scripts/scannet/scannet_val.sh
```
Evaluate the SOLE on the ScanNet200 dataset.
```
bash scripts/scannet200/scannet200_val.sh
```
If you want to use <a href="https://docs.wandb.ai" >wandb</a> during the training, set the `workspace` in `conf/config_base_instance_segmentation.yaml` file to your wandb workspace name. And run the command below before running the training/testing command:
```
wandb enabled
```
If you want to turn off the wandb, run the command below before running the training/testing command:
```
wandb disabled
```

## Acknowledgement
We build our code on top of the <a href="https://github.com/JonasSchult/Mask3D">Mask3D</a>. We sincerely thank to Mask3D team for the amazing work and well-structured code. Furthermore, our work is inspired a lot from the following works:
- <a href="https://openmask3d.github.io">OpenMask3D</a>
- <a href="https://jonasschult.github.io/Mask3D/">OpenIns3D</a>
- <a href="https://dingry.github.io/projects/PLA">PLA</a>, <a href="https://jihanyang.github.io/projects/RegionPLC">RegionPLC</a>, <a href="https://arxiv.org/abs/2308.00353">Lowis3D</a>
- <a href="https://pengsongyou.github.io/openscene">OpenScene</a>

We express our gratitude for their exceptional contributions.


## Citation
If you find our code or paper useful, please cite
```bibtex
@article{lee2024segment,
      title = {Segment Any 3D Object with Language}, 
      author = {Lee, Seungjun and Zhao, Yuyang and Lee, Gim Hee},
      year = {2024},
      journal   = {arXiv preprint arXiv:2404.02157},
}
```
