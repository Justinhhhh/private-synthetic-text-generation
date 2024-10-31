import os
from datasets import Dataset

# Create smaller val sets for fine-tuning the BERT classifiers

# phishing
phish_data_set = Dataset.from_json(os.path.join('data', 'phishing', 'train.jsonl'))
phish_data_set.class_encode_column('label').train_test_split(
    stratify_by_column='label', test_size=2000, seed=1337)['test'].to_json(
    os.path.join('data', 'phishing', 'validation.jsonl'))

# webmd
webmd_data_set = Dataset.from_json(os.path.join('data', 'webmd', 'train.jsonl'))
webmd_data_set.class_encode_column('label').train_test_split(
    stratify_by_column='label', test_size=2000, seed=1337)['test'].to_json(
    os.path.join('data', 'webmd', 'validation.jsonl'))

# thumbs-up
thumbs_set = Dataset.from_json(os.path.join('data', 'thumbs-up', 'validation.jsonl'))
os.rename(os.path.join('data', 'thumbs-up', 'validation.jsonl'), os.path.join('data', 'thumbs-up', 'orig_validation.jsonl'))
thumbs_set.class_encode_column('label').train_test_split(
    stratify_by_column='label', test_size=2000, seed=1337)['test'].to_json(
    os.path.join('data', 'thumbs-up', 'validation.jsonl'))

# swmh
swmh_set = Dataset.from_json(os.path.join('data', 'swmh', 'val.jsonl'))
swmh_set.class_encode_column('label').train_test_split(
    stratify_by_column='label', test_size=2000, seed=1337)['test'].to_json(
    os.path.join('data', 'swmh', 'validation.jsonl'))
