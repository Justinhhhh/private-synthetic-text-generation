cd baselines

# phishing

python inference.py --dataset phishing --model_path ../models/phishing/eps3/bloom/051358
python inference.py --dataset phishing --model_path ../models/phishing/eps8/bloom/051420
python inference.py --dataset phishing --model_path ../models/phishing/np/bloom/010544

# swmh

python inference.py --dataset swmh --model_path ../models/swmh/eps3/bloom/042387
python inference.py --dataset swmh --model_path ../models/swmh/eps8/bloom/042414
python inference.py --dataset swmh --model_path ../models/swmh/np/bloom/008708

# thumbs-up

python inference.py --dataset thumbs-up --model_path ../models/thumbs-up/eps3/bloom/188396
python inference.py --dataset thumbs-up --model_path ../models/thumbs-up/eps8/bloom/188338
python inference.py --dataset thumbs-up --model_path ../models/thumbs-up/np/bloom/038668

# webmd

python inference.py --dataset webmd --model_path ../models/webmd/eps3/bloom/todo
python inference.py --dataset webmd --model_path ../models/webmd/eps8/bloom/todo
python inference.py --dataset webmd --model_path ../models/webmd/np/bloom/035344
