# BTRA
BTRA: Boosting Adversarial Defenses by Breaking Trade-offs between Robustness and Accuracy
Code for the paper BTRA: Boosting Adversarial Defenses by Breaking Trade-offs between Robustness and Accuracy.

## Environment settings and libraries we used in our experiments
### This project is tested under the following environment settings:

OS: Ubuntu 20.04.3
GPU: NVIDIA A100
Cuda: 11.1, Cudnn: v8.2
Python: 3.9.5
PyTorch: 1.8.0
Torchvision: 0.9.0
Acknowledgement
The codes are modifed based on the PyTorch implementation of Rebuffi et al., 2021.

## Requirements
Install or download AutoAttack:
pip install git+https://github.com/fra31/auto-attack
## Download 1M DDPM generated data from the official implementation of Rebuffi et al., 2021:

|Dataset	Extra |	Size	     |   Link
|CIFAR-10	  |DDPM	1M	npz|
|CIFAR-100	  |DDPM	1M	npz|
|SVHN	      |DDPM	1M	npz|

## Training Commands
### To run the KL-based baselines (with 1M DDPM generated data), an example is:
```
$python train-wa.py --data-dir 'cifar-data' \
    --log-dir 'trained_models' \
    --desc 'WRN28-10Swish_cifar10s_lr0p2_TRADES5_epoch400_bs512_fraction0p7_ls0p1' \
    --data cifar10s \
    --batch-size 512 \
    --model wrn-28-10-swish \
    --num-adv-epochs 400 \
    --lr 0.2 \
    --beta 5.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename 'cifar10_ddpm.npz' \
    --ls 0.1
 ```  
Here --ls 0.1 is inherent from the the code implementation of Rebuffi et al., 2021.

## To run our methods (with 1M DDPM generated data), an example is:
```
python train-wa.py --data-dir 'cifar-data' \
    --log-dir 'trained_models' \
    --desc 'WRN28-10Swish_cifar10s_lr0p2_TRADES4_epoch400_bs512_fraction0p7_LSE' \
    --data cifar10s \
    --batch-size 512 \
    --model wrn-28-10-swish \
    --num-adv-epochs 10 \
    --lr 0.005 \
    --beta 4.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename 'cifar10_ddpm.npz' \
    --LSE --ls 0
```
Here we only need to activate the flag --LSE and set --ls 0.

## Pretrained checkpoints
Below are pretrained checkpoints of WRN-28-10 (Swish) and WRN-70-16 (Swish) with --beta=3.0:

Dataset	Model	Clean	AA		
CIFAR-10	WRN-28-10	88.61	61.04	checkpoint	argtxt
CIFAR-10	WRN-70-16	89.01	63.35	checkpoint	argtxt
CIFAR-100	WRN-28-10	63.66	31.08	checkpoint	argtxt
CIFAR-100	WRN-70-16	65.56	33.05	checkpoint	argtxt

Downloading checkpoint to trained_models/mymodel/weights-best.pt
Downloading argtxt to trained_models/mymodel/args.txt
## Evaluation Commands
For evaluation under AutoAttack, run the command (taking our method as an example):

python eval-aa.py --data-dir 'cifar-data' \
    --log-dir 'trained_models' \
    --desc mymodel

python eval-adv.py --data-dir 'cifar-data' \
    --log-dir 'trained_models' \
    --desc mymodel

python eval-muti-adv.py --data-dir 'cifar-data' \
    --log-dir 'trained_models' \
    --desc mymodel
