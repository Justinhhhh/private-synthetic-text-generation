#!/bin/bash
#
#SBATCH --job-name=phishing-results
#SBATCH --output=/ukp-storage-1/ochs/phishing-results.txt
#SBATCH --account=ukp-researcher
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

# phishing
python evaluation.py --data_file results/phishing/eps3/bart/samples.jsonl --dataset phishing
python evaluation.py --data_file results/phishing/eps8/bart/samples.jsonl --dataset phishing
python evaluation.py --data_file results/phishing/np/bart/samples.jsonl --dataset phishing

python evaluation.py --data_file results/phishing/eps3/bloom/samples.jsonl --dataset phishing
python evaluation.py --data_file results/phishing/eps8/bloom/samples.jsonl --dataset phishing
python evaluation.py --data_file results/phishing/np/bloom/samples.jsonl --dataset phishing

python evaluation.py --data_file results/phishing/eps3/phi/samples.jsonl --dataset phishing
python evaluation.py --data_file results/phishing/eps8/phi/samples.jsonl --dataset phishing
python evaluation.py --data_file results/phishing/np/phi/samples.jsonl --dataset phishing

python evaluation.py --data_file results/phishing/eps3/diffuseq/samples.jsonl --dataset phishing
python evaluation.py --data_file results/phishing/eps8/diffuseq/samples.jsonl --dataset phishing
python evaluation.py --data_file results/phishing/np/diffuseq/samples.jsonl --dataset phishing

python evaluation.py --data_file results/phishing/eps3/seqdiffuseq/samples.jsonl --dataset phishing
python evaluation.py --data_file results/phishing/eps8/seqdiffuseq/samples.jsonl --dataset phishing
python evaluation.py --data_file results/phishing/np/seqdiffuseq/samples.jsonl --dataset phishing

python evaluation.py --data_file results/phishing/eps3/genie/samples.jsonl --dataset phishing
python evaluation.py --data_file results/phishing/eps8/genie/samples.jsonl --dataset phishing
python evaluation.py --data_file results/phishing/np/genie/samples.jsonl --dataset phishing

:'
# swmh
python evaluation.py --data_file results/swmh/eps3/bart/samples.jsonl --dataset swmh
python evaluation.py --data_file results/swmh/eps8/bart/samples.jsonl --dataset swmh
python evaluation.py --data_file results/swmh/np/bart/samples.jsonl --dataset swmh

python evaluation.py --data_file results/swmh/eps3/bloom/samples.jsonl --dataset swmh
python evaluation.py --data_file results/swmh/eps8/bloom/samples.jsonl --dataset swmh
python evaluation.py --data_file results/swmh/np/bloom/samples.jsonl --dataset swmh

python evaluation.py --data_file results/swmh/eps3/phi/samples.jsonl --dataset swmh
python evaluation.py --data_file results/swmh/eps8/phi/samples.jsonl --dataset swmh
python evaluation.py --data_file results/swmh/np/phi/samples.jsonl --dataset swmh

python evaluation.py --data_file results/swmh/eps3/diffuseq/samples.jsonl --dataset swmh
python evaluation.py --data_file results/swmh/eps8/diffuseq/samples.jsonl --dataset swmh
python evaluation.py --data_file results/swmh/np/diffuseq/samples.jsonl --dataset swmh

python evaluation.py --data_file results/swmh/eps3/seqdiffuseq/samples.jsonl --dataset swmh
python evaluation.py --data_file results/swmh/eps8/seqdiffuseq/samples.jsonl --dataset swmh
python evaluation.py --data_file results/swmh/np/seqdiffuseq/samples.jsonl --dataset swmh

python evaluation.py --data_file results/swmh/eps3/genie/samples.jsonl --dataset swmh
python evaluation.py --data_file results/swmh/eps8/genie/samples.jsonl --dataset swmh
python evaluation.py --data_file results/swmh/np/genie/samples.jsonl --dataset swmh


# thumbs-up
python evaluation.py --data_file results/thumbs-up/eps3/bart/samples.jsonl --dataset thumbs-up
python evaluation.py --data_file results/thumbs-up/eps8/bart/samples.jsonl --dataset thumbs-up
python evaluation.py --data_file results/thumbs-up/np/bart/samples.jsonl --dataset thumbs-up

python evaluation.py --data_file results/thumbs-up/eps3/bloom/samples.jsonl --dataset thumbs-up
python evaluation.py --data_file results/thumbs-up/eps8/bloom/samples.jsonl --dataset thumbs-up
python evaluation.py --data_file results/thumbs-up/np/bloom/samples.jsonl --dataset thumbs-up

python evaluation.py --data_file results/thumbs-up/eps3/phi/samples.jsonl --dataset thumbs-up
python evaluation.py --data_file results/thumbs-up/eps8/phi/samples.jsonl --dataset thumbs-up
python evaluation.py --data_file results/thumbs-up/np/phi/samples.jsonl --dataset thumbs-up

python evaluation.py --data_file results/thumbs-up/eps3/diffuseq/samples.jsonl --dataset thumbs-up
python evaluation.py --data_file results/thumbs-up/eps8/diffuseq/samples.jsonl --dataset thumbs-up
python evaluation.py --data_file results/thumbs-up/np/diffuseq/samples.jsonl --dataset thumbs-up

python evaluation.py --data_file results/thumbs-up/eps3/seqdiffuseq/samples.jsonl --dataset thumbs-up
python evaluation.py --data_file results/thumbs-up/eps8/seqdiffuseq/samples.jsonl --dataset thumbs-up
python evaluation.py --data_file results/thumbs-up/np/seqdiffuseq/samples.jsonl --dataset thumbs-up

python evaluation.py --data_file results/thumbs-up/eps3/genie/samples.jsonl --dataset thumbs-up
python evaluation.py --data_file results/thumbs-up/eps8/genie/samples.jsonl --dataset thumbs-up
python evaluation.py --data_file results/thumbs-up/np/genie/samples.jsonl --dataset thumbs-up

# webmd

python evaluation.py --data_file results/webmd/eps3/bart/samples.jsonl --dataset webmd
python evaluation.py --data_file results/webmd/eps8/bart/samples.jsonl --dataset webmd
python evaluation.py --data_file results/webmd/np/bart/samples.jsonl --dataset webmd

python evaluation.py --data_file results/webmd/eps3/bloom/samples.jsonl --dataset webmd
python evaluation.py --data_file results/webmd/eps8/bloom/samples.jsonl --dataset webmd
python evaluation.py --data_file results/webmd/np/bloom/samples.jsonl --dataset webmd

python evaluation.py --data_file results/webmd/eps3/phi/samples.jsonl --dataset webmd
python evaluation.py --data_file results/webmd/eps8/phi/samples.jsonl --dataset webmd
python evaluation.py --data_file results/webmd/np/phi/samples.jsonl --dataset webmd

python evaluation.py --data_file results/webmd/eps3/diffuseq/samples.jsonl --dataset webmd
python evaluation.py --data_file results/webmd/eps8/diffuseq/samples.jsonl --dataset webmd
python evaluation.py --data_file results/webmd/np/diffuseq/samples.jsonl --dataset webmd

python evaluation.py --data_file results/webmd/eps3/seqdiffuseq/samples.jsonl --dataset webmd
python evaluation.py --data_file results/webmd/eps8/seqdiffuseq/samples.jsonl --dataset webmd
python evaluation.py --data_file results/webmd/np/seqdiffuseq/samples.jsonl --dataset webmd

python evaluation.py --data_file results/webmd/eps3/genie/samples.jsonl --dataset webmd
python evaluation.py --data_file results/webmd/eps8/genie/samples.jsonl --dataset webmd
python evaluation.py --data_file results/webmd/np/genie/samples.jsonl --dataset webmd
'
