#!/bin/bash
#
#SBATCH --job-name=diffuseq
#SBATCH --output=/ukp-storage-1/ochs/diffuseq.txt
#SBATCH --account=ukp-researcher
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1

cd DiffuSeq

python train.py --data_dir ../data/phishing --dataset phishing --learning_steps 20000 --batch_size 32
python train.py --data_dir ../data/phishing --dataset phishing --learning_steps 20000 --batch_size 32 --private True --epsilon 1
python train.py --data_dir ../data/phishing --dataset phishing --learning_steps 20000 --batch_size 32 --private True --epsilon 3
python train.py --data_dir ../data/phishing --dataset phishing --learning_steps 20000 --batch_size 32 --private True --epsilon 8

python train.py --data_dir ../data/webmd --dataset webmd --learning_steps 20000 --batch_size 32
python train.py --data_dir ../data/webmd --dataset webmd --learning_steps 20000 --batch_size 32 --private True --epsilon 1
python train.py --data_dir ../data/webmd --dataset webmd --learning_steps 20000 --batch_size 32 --private True --epsilon 3
python train.py --data_dir ../data/webmd --dataset webmd --learning_steps 20000 --batch_size 32 --private True --epsilon 8

python train.py --data_dir ../data/thumbs-up --dataset thumbs-up --learning_steps 20000 --batch_size 32
python train.py --data_dir ../data/thumbs-up --dataset thumbs-up --learning_steps 20000 --batch_size 32 --private True --epsilon 1
python train.py --data_dir ../data/thumbs-up --dataset thumbs-up --learning_steps 20000 --batch_size 32 --private True --epsilon 3
python train.py --data_dir ../data/thumbs-up --dataset thumbs-up --learning_steps 20000 --batch_size 32 --private True --epsilon 8

