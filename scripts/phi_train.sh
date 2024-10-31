cd baselines

# np
python training.py --dataset phishing --epochs 4 --batch_size 8 --model phi
python training.py --dataset swmh --epochs 4 --batch_size 8 --model phi
python training.py --dataset thumbs-up --epochs 4 --batch_size 8 --model phi
python training.py --dataset webmd --epochs 4 --batch_size 8 --model phi

# e3
python training.py --dataset phishing --epochs 8 --batch_size 8 --private True --epsilon 3 --model phi
python training.py --dataset swmh --epochs 8 --batch_size 8 --private True --epsilon 3 --model phi
python training.py --dataset thumbs-up --epochs 8 --batch_size 8 --private True --epsilon 3 --model phi
python training.py --dataset webmd --epochs 8 --batch_size 8 --private True --epsilon 3 --model phi

# e8
python training.py --dataset phishing --epochs 8 --batch_size 8 --private True --epsilon 8 --model phi
python training.py --dataset swmh --epochs 8 --batch_size 8 --private True --epsilon 8 --model phi
python training.py --dataset thumbs-up --epochs 8 --batch_size 8 --private True --epsilon 8 --model phi
python training.py --dataset webmd --epochs 8 --batch_size 8 --private True --epsilon 8 --model phi

