<div align="center">
<h1 align="center">SPMNet</h1>

<h3>SPMNet: A Siamese Pyramid Mamba Network for Very-High-Resolution Remote Sensing Change Detection</h3>

[Jiashu Wang]<sup>1</sup>, [Jinze Song]<sup>1</sup>, [Hao Zhang]<sup>1</sup>, [Zekai Zhang]<sup>2</sup>, [Yunlong Ji]<sup>1</sup> [Wenyin Zhang]<sup>1*</sup>, [Xing Wang]<sup>1</sup>, [Jinglin Zhang]<sup>1,2</sup>

<sup>1</sup> The University of Linyi, <sup>2</sup> Shandong University

<sup>*</sup> Corresponding author




</div>

## 🛎️Updates
* **` Notice🐍🐍`**: SPMNet has been accepted by [IEEE TGRS](https://ieeexplore.ieee.org/document/10981441)! We'd appreciate it if you could give this repo a ⭐️**star**⭐️ and stay tuned!!
* **` June 17th, 2024`**: SPMNte has been accepted by [IEEE TGRS](https://ieeexplore.ieee.org/document/10981441)!!


## 🔭Overview

Very-high-resolution (VHR) remote sensing images are characterized by extremely high-spatial resolution, incorporating higher pixel density and larger image sizes, which pose challenges for existing methods to extract complex texture features. Furthermore, due to the wide-area and high-resolution imaging strategy, VHR change detection images suffer from a severe imbalance between change pixels and nonchange pixels, increasing the difficulty of handling change detection tasks. To address these challenges, we introduced the omnidirectional selective scan module (OSSM), which has the capability to process long sequences. By integrating it with the lightweight Siamese feature pyramid network (SFPN), we designed a hybrid CNN-Mamba backbone, referred to as Siamese pyramid Mamba (SPMamba). This backbone captures both global and local information within bitemporal feature maps at each stage, enhancing the precision of texture feature extraction. In addition, to integrate the semantic features from each branch in SPMamba and reduce noise interference from nontarget change areas, we developed a hybrid fusion module (HFM). The HFM consists of two fusion modules: the high- and low-channel fusion module (HLM) and the bilateral channel fusion module (BCM), which facilitates both feature-level and channel-level integration, enhancing the sensitivity of the model to subtle changes. Extensive experimental results demonstrate that SPMNet achieves the highest F1-score of 91.80%, 90.99%, and 96.04% on the WHU-CD, LEVIR-CD, and CDD-CD datasets, respectively, outperforming 11 state-of-the-art methods. Moreover, the effects of varying image sizes on model training are thoroughly analyzed.


## 🗝️Let's Get Started!
### `A. Installation`

Note that the code in this repo runs under **Linux** system. We have not tested whether it works under other OS.

The repo is based on the [VMamba repo](https://github.com/MzeroMiko/VMamba), thus you need to install it first. The following installation sequence is taken from the VMamba repo. 

**Step 1: Clone the repository:**

Clone this repository and navigate to the project directory:
```bash
git clone https://github.com/0130wangjiashu/SPMNet.git
cd SPMNet
```


**Step 2: Environment Setup:**

It is recommended to set up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:

***Create and activate a new conda environment***

```bash
conda create -n SPMNet
conda activate SPMNet
```

***Install dependencies***

```bash
pip install -r requirements.txt
```

### `B. Data Preparation`
***Binary change detection***

All the VHR change detection datasets [CDD], [LEVIR\LEVIR-CD+](https://chenhao.in/LEVIR/) and [WHU-CD](http://gpcv.whu.edu.cn/data/building_dataset.html) are used for binary change detection experiments. Please download them and make them have the following folder/file structure:
```
${DATASET_ROOT} # dataset root dir
├── train
    ├── t1
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── t2
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── label
        └── 0001.tif
        └── 0002.tif
        └── ...
├── val
    ├── t1
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── t2
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── label
        ├── 0001.tif
        └── 0002.tif
        └── ...
├── test
    ├── t1
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── t2
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── label
        └── 0001.tif
        └── 0002.tif
        └── ...
```




### `C. Training and Inference of Change Detection Model`
To start training, run the following code in the command line: 

```
python train.py
python test.py
```


## 📜Reference

If this code or dataset contributes to your research, please kindly consider citing our paper and give this repo ⭐️ :)
```
@ARTICLE{10981441,
  author={Wang, Jiashu and Song, Jinze and Zhang, Hao and Zhang, Zekai and Ji, Yunlong and Zhang, Wenyin and Zhang, Jinglin and Wang, Xing},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SPMNet: A Siamese Pyramid Mamba Network for Very-High-Resolution Remote Sensing Change Detection}, 
  year={2025},
  volume={63},
  number={},
  pages={1-14},
  keywords={Remote sensing;Feature extraction;Computational modeling;Transformers;Deep learning;Convolutional neural networks;Visualization;Semantics;Electronic mail;Training;Change detection;hybrid fusion module (HFM);Siamese pyramid Mamba (SPMamba);very-high-resolution (VHR)},
  doi={10.1109/TGRS.2025.3565801}}
```



## 🤝Acknowledgments
This project is based on VMamba ([paper](https://arxiv.org/abs/2401.10166), [code](https://github.com/MzeroMiko/VMamba)), ScanNet [code](https://github.com/NJU-LHRS/Official_Remote_Sensing_Mamba). Thanks for their excellent works!!
