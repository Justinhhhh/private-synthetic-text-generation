cd baselines

# np
python training.py --dataset phishing --epochs 4 --batch_size 16 --model bloom
python training.py --dataset swmh --epochs 4 --batch_size 16 --model bloom
python training.py --dataset thumbs-up --epochs 4 --batch_size 16 --model bloom
python training.py --dataset webmd --epochs 4 --batch_size 16 --model bloom

# e3
python training.py --dataset phishing --epochs 8 --batch_size 16 --private True --epsilon 3 --model bloom
python training.py --dataset swmh --epochs 8 --batch_size 16 --private True --epsilon 3 --model bloom
python training.py --dataset thumbs-up --epochs 8 --batch_size 16 --private True --epsilon 3 --model bloom
python training.py --dataset webmd --epochs 8 --batch_size 16 --private True --epsilon 3 --model bloom

# e8
python training.py --dataset phishing --epochs 8 --batch_size 16 --private True --epsilon 8 --model bloom
python training.py --dataset swmh --epochs 8 --batch_size 16 --private True --epsilon 8 --model bloom
python training.py --dataset thumbs-up --epochs 8 --batch_size 16 --private True --epsilon 8 --model bloom
python training.py --dataset webmd --epochs 8 --batch_size 16 --private True --epsilon 8 --model bloom

