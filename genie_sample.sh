cd GENIE

# phishing
python Genie_Generate.py --model_path models/phishing/eps3/56008 --dataset phishing
python Genie_Generate.py --model_path models/phishing/eps8/56008 --dataset phishing
python Genie_Generate.py --model_path models/phishing/np/50160 --dataset phishing

#swmh
python Genie_Generate.py --model_path models/swmh/eps3/55899 --dataset swmh
python Genie_Generate.py --model_path models/swmh/eps8/55899 --dataset swmh


#thumbs-up
python Genie_Generate.py --model_path models/thumbs-up/eps3/56871 --dataset thumbs-up
python Genie_Generate.py --model_path models/thumbs-up/eps8/56871 --dataset thumbs-up
python Genie_Generate.py --model_path models/thumbs-up/np/50778 --dataset thumbs-up

#webmd
python Genie_Generate.py --model_path models/webmd/eps3/56986 --dataset webmd
python Genie_Generate.py --model_path models/webmd/eps8/56986 --dataset webmd
python Genie_Generate.py --model_path models/webmd/np/50830 --dataset webmd
