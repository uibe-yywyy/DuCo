This repository gives the implementation of our proposed method DuCo.

Requirements:
matplotlib==3.5.2
numpy==1.22.3
Pillow==9.4.0
scikit_learn==1.2.2
scipy==1.8.0
seaborn==0.11.2
tensorboard_logger==0.1.0
torch==1.13.0
torchvision==0.14.0


Demo:

python main.py --dataset cifar10  --epochs 500 --batch-size 256 --lr 0.1 --lam 1 --proto_m 0.99 --tau_proto 1

python main.py --dataset svhn  --epochs 500 --batch-size 256 --lr 0.1 --lam 1 --proto_m 0.99 --tau_proto 1
