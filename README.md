<div align="center">
<h1 align="center">HAM-CD</h1>

<h3>HAM-CD: Hybrid Attention Mamba for Remote Sensing Change Detection</h3>

## 🗝️Let's Get Started!
### `A. Installation`

Note that the code in this repo runs under **Linux** system. We have not tested whether it works under other OS.

The repo is based on the [VMama repo](https://github.com/MzeroMiko/VMamba), thus you need to install it first. The following installation sequence is taken from the VMamba repo. 

**Step 1: Clone the repository:**

Clone this repository and navigate to the project directory:
```bash
git clone https://github.com/ChenHongruixuan/HAM-CD.git
cd HAM-CD
```


**Step 2: Environment Setup:**

It is recommended to set up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:

***Create and activate a new conda environment***

```bash
conda create -n ham-cd
conda activate ham-cd
```

***Install dependencies***

```bash
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```


***Dependencies for "Detection" and "Segmentation" (optional in VMamba)***

```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```
### `B. Download Pretrained Weight`
Also, please download the pretrained weights of [VMamba-Tiny](https://zenodo.org/records/14037769), [VMamba-Small](https://zenodo.org/records/14037769), and [VMamba-Base](https://zenodo.org/records/14037769) and put them under 
```bash
project_path/MambaCD/pretrained_weight/
```

### `C. Data Preparation`
***Binary change detection***

The three datasets [SYSU](https://github.com/liumency/SYSU-CD), [LEVIR-CD+](https://chenhao.in/LEVIR/) and [WHU-CD](http://gpcv.whu.edu.cn/data/building_dataset.html) are used for binary change detection experiments. Please download them and make them have the following folder/file structure:
```
${DATASET_ROOT}   # Dataset root directory, for example: /home/username/data/SYSU
├── train
│   ├── T1
│   │   ├──00001.png
│   │   ├──00002.png
│   │   ├──00003.png
│   │   ...
│   │
│   ├── T2
│   │   ├──00001.png
│   │   ... 
│   │
│   └── GT
│       ├──00001.png 
│       ...   
│   
├── test
│   ├── ...
│   ...
│  
├── train.txt   # Data name list, recording all the names of training data
└── test.txt    # Data name list, recording all the names of testing data
```

### `D. Model Training`
Before training models, please enter into [`changedetection`] folder, which contains all the code for network definitions, training and testing. 

```bash
cd <project_path>/HAM-CD/changedetection
```

***Binary change detection***

The following commands show how to train and evaluate MambaBCD-Small on the SYSU dataset:
```bash
python script/train_HAMBCD.py  --dataset 'SYSU'  --batch_size 8   --crop_size 256   --max_iters 320000   --model_type MambaBCD_Small  --model_param_path '/data/lgl/codes/MambaCD/changedetection/saved_models'    --train_dataset_path '/data/lgl/datasets/SYSU-CD/train'  --train_data_list_path '/data/lgl/datasets/SYSU-CD/train_list.txt'    --test_dataset_path '/data/lgl/datasets/SYSU-CD/test'   --test_data_list_path '/data/lgl/datasets/SYSU-CD/test_list.txt'     --cfg '/data/lgl/codes/MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml'  --pretrained_weight_path '/data/lgl/codes/MambaCD/pretrained_weight/vssm_small_0229_ckpt_epoch_222.pth'
```

### `E. Inference Using Our/Your Weights`

Before inference, please enter into [`changedetection`] folder. 
```bash
cd <project_path>/HAM-CD/changedetection
```


***Binary change detection***

The following commands show how to infer binary change maps using trained MambaBCD-Tiny on the LEVIR-CD+ dataset:

* **` Kind reminder`**: Please use [--resume] to load our trained model, instead of using [--pretrained_weight_path]. 

```bash
python script/infer_HAMBCD.py  --dataset 'SYSU' --model_type 'MambaBCD_Small'  --test_dataset_path '/data/lgl/datasets/SYSU-CD/test'  --test_data_list_path '/data/lgl/datasets/SYSU-CD/test_list.txt' --cfg '/data/lgl/codes/MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml'  --resume '/data/lgl/codes/MambaCD/changedetection/saved_models/SYSU/MambaBCD_Small_1742872326.241806/40000_model.pth'
```
