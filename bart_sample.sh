cd baselines

# phishing

python inference.py --dataset phishing --model_path ../models/phishing/eps3/bart/046827
python inference.py --dataset phishing --model_path ../models/phishing/eps8/bart/046887
python inference.py --dataset phishing --model_path ../models/phishing/np/bart/005272
# swmh

python inference.py --dataset swmh --model_path ../models/swmh/eps3/bart/038532
python inference.py --dataset swmh --model_path ../models/swmh/eps8/bart/038488
python inference.py --dataset swmh --model_path ../models/swmh/np/bart/004356
# thumbs-up

python inference.py --dataset thumbs-up --model_path ../models/thumbs-up/eps3/bart/171689
python inference.py --dataset thumbs-up --model_path ../models/thumbs-up/eps8/bart/171707
python inference.py --dataset thumbs-up --model_path ../models/thumbs-up/np/bart/019336

# webmd
python inference.py --dataset webmd --model_path ../models/webmd/eps3/bart/156678
python inference.py --dataset webmd --model_path ../models/webmd/eps8/bart/156837
python inference.py --dataset webmd --model_path ../models/webmd/np/bart/017672



