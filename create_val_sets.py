import datasets

# phishing
phish_data_set = datasets.Dataset.from_json('data/phishing/train.jsonl')
phish_data_set.class_encode_column('label').train_test_split(
    stratify_by_column='label', test_size=2000, seed=1337)['test'].to_json('data/phishing/validation.jsonl')

#webmd
webmd_data_set = datasets.Dataset.from_json('data/webmd/train.jsonl')
webmd_data_set.class_encode_column('label').train_test_split(
    stratify_by_column='label', test_size=2000, seed=1337)['test'].to_json('data/webmd/validation.jsonl')

#thumbs-up
thumbs_set = datasets.Dataset.from_json('data/thumbs-up/original_validation.jsonl')
thumbs_set.class_encode_column('label').train_test_split(
    stratify_by_column='label', test_size=2000, seed=1337)['test'].to_json('data/thumbs-up/validation.jsonl')

#swmh
swmh_set = datasets.Dataset.from_json('data/swmh/original_validation.jsonl')
swmh_set.class_encode_column('label').train_test_split(
    stratify_by_column='label', test_size=2000, seed=1337)['test'].to_json('data/swmh/validation.jsonl')
