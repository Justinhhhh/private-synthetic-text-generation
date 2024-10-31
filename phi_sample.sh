#!/bin/bash
#
#SBATCH --job-name=phi-e8
#SBATCH --output=/ukp-storage-1/ochs/phi-e8.txt
#SBATCH --account=ukp-researcher
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

cd baselines

# phishing
python inference.py --dataset phishing --model_path ../models/phishing/np/phi/021088
python inference.py --dataset phishing --model_path ../models/phishing/eps3/phi/092203
python inference.py --dataset phishing --model_path ../models/phishing/eps8/phi/092408

# swmh
python inference.py --dataset swmh --model_path ../models/swmh/np/phi/017412
python inference.py --dataset swmh --model_path ../models/swmh/eps3/phi/076115
python inference.py --dataset swmh --model_path ../models/swmh/eps8/phi/076113

# thumbs-up
python inference.py --dataset thumbs-up --model_path ../models/thumbs-up/np/phi/077336
python inference.py --dataset thumbs-up --model_path ../models/thumbs-up/eps3/phi/todo
python inference.py --dataset thumbs-up --model_path ../models/thumbs-up/eps8/phi/todo

# webmd
python inference.py --dataset webmd --model_path ../models/webmd/np/phi/070684
python inference.py --dataset webmd --model_path ../models/webmd/eps3/phi/todo
python inference.py --dataset webmd --model_path ../models/webmd/eps8/phi/todo