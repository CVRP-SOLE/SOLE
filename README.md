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
  <h2 align="center">ICLR 2025</h2>
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
      <a href="#download-data-and-weight">Download data and weight</a>
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

- [2024/04/20] Code is released 💡.
- [2024/05/02] Pre-processed data and weights are released. Now you can train and evaluate our SOLE 👏🏻.
- [2025/01/23] SOLE is accepted to ICLR 2025 🔥. The code for Replica dataset and preprocessing the MMA features will be released soon.

## TODO
- [x] Release the code
- [x] Release the preprocessed data and weights
- [ ] Release the evaluation code for Replica dataset
- [ ] Release the pre-processed data and precomputed features for Replica dataset

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

We provide the **pre-processed 3D data** and **precomputed features** for the training and evaluation which are listed below:
- Pre-processed 3D data
  * <a href="https://huggingface.co/datasets/onandon/SOLE/tree/main/processed/scannet">ScanNet</a>
  * <a href="https://huggingface.co/datasets/onandon/SOLE/tree/main/processed/scannet200">ScanNet200</a>
  * ~~Replica~~ (coming soon)
- Precomputed per-point CLIP features
  * <a href="https://huggingface.co/datasets/onandon/SOLE/tree/main/openseg/scannet">ScanNet</a>
  * ~~Replica~~ (coming soon)
- Precomputed features of MCA and MEA
  * ScanNet : <a href="https://huggingface.co/datasets/onandon/SOLE/tree/main/scannet_mca">MCA</a>, <a href="https://huggingface.co/datasets/onandon/SOLE/tree/main/scannet_mea">MEA</a>
  * ScanNet200 : <a href="https://huggingface.co/datasets/onandon/SOLE/tree/main/scannet200_mca">MCA</a>, <a href="https://huggingface.co/datasets/onandon/SOLE/tree/main/scannet200_mea">MEA</a> 

You can download above data with following <a href="#download-data-and-weight">Download data and weight</a>. We also provide the specific data configuration in <a href="https://huggingface.co/datasets/onandon/SOLE">here</a> to help your understanding for our pre-processed data.

## Weights

For the stable training, we employ a two-stage training process:

1. Pretrain the backbone with only using mask-annotations.
2. Train the mask decoder while backbone is fixed. Mask annotations and three types of associations are used for the training. (See the original paper for the details.)

For the training, we provide pretrained backbone weights for ScanNet and ScanNet200 datasets listed below:
- <a href="https://huggingface.co/datasets/onandon/SOLE/blob/main/backbone_scannet.ckpt">Backbone weights for ScanNet</a>
- <a href="https://huggingface.co/datasets/onandon/SOLE/blob/main/backbone_scannet200.ckpt">Backbone weights for ScanNet200</a>

For the evaluation, we provide the official weight of SOLE for ScanNet and ScanNet200 datasets listed below:
- <a href="https://huggingface.co/datasets/onandon/SOLE/blob/main/scannet.ckpt">Offical weights of SOLE for ScanNet</a>
- <a href="https://huggingface.co/datasets/onandon/SOLE/blob/main/scannet200.ckpt">Official weights of SOLE for ScanNet200</a>
- ~~Official weights of SOLE for Replica~~ (coming soon)

You can download all of the weights for the pretrained backbone and SOLE with following <a href="#download-data-and-weight">Download data and weight</a>.

## Download data and weight

We provide the python script that download all of the pre-processed data and weights we mentioned above. You can run the command below:
```
python download_data.py
```

Once you run the above command, the downloaded files must be automatically located to the corresponding path. Refer to the file structure below.

```
├── backbone_checkpoint
│   ├── backbone_scannet.ckpt        <- Backbone weights for ScanNet
│   └── backbone_scannet200.ckpt     <- Backobne weights for ScanNet200
│
├── checkpoint
│   ├── scannet.ckpt        <- Official weights for ScanNet
│   └── scannet200.ckpt     <- Official weights for ScanNet200
│ 
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

If you successfully download all of the given files, you are now ready to train and evaluate the model. Check the training and evaluation command in <a href="#training-and-testing">Training and Testing</a> section to run the SOLE.

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
