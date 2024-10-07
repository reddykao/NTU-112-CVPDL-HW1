# NTU-CVPDL-HW1
## Installation
### My Environment
* CPU: Intel(R) Xeon(R) W-3223 CPU @ 3.50GHz
* GPU: NVIDIA RTX 3090
* OS: Ubuntu 22.04.2 LTS
* NVIDIA Driver Version: 525.125.06
* NVIDIA CUDA Version: 12.0

### Requirements
* Python 3.10.12
* torch 2.0.1+cu118
* torchvision 0.15.2+cu118
* numpy 1.23.5
* Pillow  9.0.1
* tqdm 4.65.2
## Usuage
```
code_directory/ train.py, valid.py, test.py
└── hw1_dataset/
    ├── train/
    |     └── train.json
    ├── valid/
    |     └── val.json
    ├── test/
    |     └── test.json
    └── annotations/
        	├── train.json
        	└── val.json
```
### Dataset preparation
### Training ( You can switch def-detr or detr by --arch)  
```
python3 train.py --arch def-detr -p /code directory/hw1_dataset --epochs 810
```
### Evaluate (set epoch randomly because it will skip training automatically by setting max_epochs < your ckpt epochs )
```
python3 eva.py --arch def-detr -p /code directory/hw1_dataset --epochs 809
```
### Test (set epoch randomly because it will skip training automatically by setting max_epochs < your ckpt epochs )
```
python3 test.py --arch def-detr -p /code directory/hw1_dataset --epochs 809
```

