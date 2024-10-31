cd DiffuSeq

# phishing
python decode.py --model_path ../models/phishing/eps3/400000.pt
python decode.py --model_path ../models/phishing/eps8/400000.pt
python decode.py --model_path ../models/phishing/np/200000.pt

# swmh
python decode.py --model_path ../models/swmh/eps3/400000.pt
python decode.py --model_path ../models/swmh/eps8/400000.pt
python decode.py --model_path ../models/swmh/np/200000.pt

# thumbs-up
python decode.py --model_path ../models/thumbs-up/eps3/400000.pt
python decode.py --model_path ../models/thumbs-up/eps8/400000.pt
python decode.py --model_path ../models/thumbs-up/np/200000.pt

# webmd
python decode.py --model_path ../models/webmd/eps3/400000.pt
python decode.py --model_path ../models/webmd/eps8/400000.pt
python decode.py --model_path ../models/webmd/np/200000.pt
