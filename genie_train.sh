cd GENIE

# np
python Genie_Finetune.py --dataset phishing --batch_size 128
python Genie_Finetune.py --dataset swmh --batch_size 128
python Genie_Finetune.py --dataset thumbs-up --batch_size 128
python Genie_Finetune.py --dataset webmd --batch_size 128

# e3
python Genie_Finetune.py --dataset phishing --batch_size 256 --private True --epsilon 3
python Genie_Finetune.py --dataset swmh --batch_size 256 --private True --epsilon 3
python Genie_Finetune.py --dataset thumbs-up --batch_size 256 --private True --epsilon 3
python Genie_Finetune.py --dataset webmd --batch_size 256 --private True --epsilon 3

# e8
python Genie_Finetune.py --dataset phishing --batch_size 256 --private True --epsilon 8
python Genie_Finetune.py --dataset swmh --batch_size 256 --private True --epsilon 8
python Genie_Finetune.py --dataset thumbs-up --batch_size 256 --private True --epsilon 8
python Genie_Finetune.py --dataset webmd --batch_size 256 --private True --epsilon 8
