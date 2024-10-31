#!/bin/bash
#
#SBATCH --job-name=bart-e8
#SBATCH --output=/ukp-storage-1/ochs/bart-e8.txt
#SBATCH --account=ukp-researcher
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1

cd baselines

# np
# python training.py --dataset phishing --epochs 4 --batch_size 32 --model bart
# python training.py --dataset swmh --epochs 4 --batch_size 32 --model bart
# python training.py --dataset thumbs-up --epochs 4 --batch_size 32 --model bart
# python training.py --dataset webmd --epochs 4 --batch_size 32 --model bart

# e3
# python training.py --dataset phishing --epochs 8 --batch_size 32 --private True --epsilon 3 --model bart
# python training.py --dataset swmh --epochs 8 --batch_size 32 --private True --epsilon 3 --model bart
# python training.py --dataset thumbs-up --epochs 8 --batch_size 32 --private True --epsilon 3 --model bart
# python training.py --dataset webmd --epochs 8 --batch_size 32 --private True --epsilon 3 --model bart

# e8
python training.py --dataset phishing --epochs 8 --batch_size 32 --private True --epsilon 8 --model bart
python training.py --dataset swmh --epochs 8 --batch_size 32 --private True --epsilon 8 --model bart
python training.py --dataset thumbs-up --epochs 8 --batch_size 32 --private True --epsilon 8 --model bart
python training.py --dataset webmd --epochs 8 --batch_size 32 --private True --epsilon 8 --model bart

