cd GENIE

# phishing

python Genie_Finetune.py --dataset psytar --batch_size 256 --private True --epsilon 3
python Genie_Generate.py --model_path models/phishing/eps3/50000 --dataset psytar
