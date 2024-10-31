import json

import torch
import transformers
import datasets
import argparse
import os
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm
from collections import defaultdict
import random
import numpy as np
import evaluate

perplexity = evaluate.load("perplexity", module_type="metric")


def evaluate_synth_data(data_path, dataset_name, n, rng):
    def preprocess(data):
        def tokenize(example):
            return tokenizer(example['trg'], max_length=256, truncation=True, padding='max_length',
                             add_special_tokens=True)

        data = data.map(tokenize, remove_columns=['src', 'trg']).rename_column('label', 'labels')
        data.set_format('torch')
        return data

    torch.manual_seed(rng)
    np.random.seed(rng)
    random.seed(rng)

    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

    train_set = datasets.Dataset.from_json(data_path).select(range(n))
    val_set = datasets.Dataset.from_json(os.path.join('data', dataset_name, 'validation.jsonl'))
    test_set = datasets.Dataset.from_json(os.path.join('data', dataset_name, 'test.jsonl'))

    # hyperparameters
    bsz = 8
    lr = 2e-5
    epochs = 5

    train_loader = torch.utils.data.DataLoader(preprocess(train_set), shuffle=True, batch_size=bsz)
    val_loader = torch.utils.data.DataLoader(preprocess(val_set), shuffle=False, batch_size=bsz)
    test_loader = torch.utils.data.DataLoader(preprocess(test_set), shuffle=False, batch_size=bsz)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=len(set(train_set['label'])))
    classifier_path = os.path.join(os.path.dirname(data_path), 'best')
    results_path = os.path.join(os.path.dirname(data_path), str(rng) + '_' + str(n) + '_results.jsonl')

    model.to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # training and validating
    best = 0
    for epoch in range(epochs):
        # Training
        model.train()
        for step, batch in enumerate(tqdm(train_loader, position=0, leave=True, desc="epoch {} train: ".format(epoch))):
            batch = {k: v.squeeze().to('cuda') for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        results = defaultdict(list)
        for step, batch in enumerate(tqdm(val_loader, position=0, leave=True, desc="epoch {} val: ".format(epoch))):
            batch = {k: v.squeeze().to('cuda') for k, v in batch.items()}
            output = model(**batch)

            results['prediction'].extend(torch.argmax(output.logits, dim=-1).tolist())
            results['labels'].extend(batch['labels'].tolist())

        acc = accuracy_score(results['labels'], results['prediction'])
        mf1 = f1_score(results['labels'], results['prediction'], average='macro')
        print("Acc: ", acc, "\nMF1", mf1)
        if mf1 > best:
            model.save_pretrained(classifier_path)
            best = mf1

    # testing
    test_results = defaultdict(list)
    for step, batch in enumerate(tqdm(test_loader, position=0, leave=True, desc="testing...")):
        batch = {k: v.squeeze().to('cuda') for k, v in batch.items()}
        output = model(**batch)

        test_results['prediction'].extend(torch.argmax(output.logits, dim=-1).tolist())
        test_results['labels'].extend(batch['labels'].tolist())

    acc = accuracy_score(test_results['labels'], test_results['prediction'])
    mf1 = f1_score(test_results['labels'], test_results['prediction'], average='macro')

    train_perplexity = perplexity.compute(predictions=[x for x in train_set['trg'] if len(x) > 0],
                                          model_id='bigscience/bloom-560m',
                                          device='cuda',
                                          batch_size=64)['mean_perplexity']
    print("Test Acc: ", acc, "\nTest MF1", mf1, "\nPerplexity", train_perplexity,
          "Empty strings: ", len([s for s in train_set['trg'] if len(s) == 0]))

    with open(results_path, 'w') as f:
        f.write(json.dumps({"test_acc": acc, "mf1": mf1, "perplexity": train_perplexity}) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True, help='Name of the synth data file')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    args = parser.parse_args()
    if args.dataset == 'drugs':
        for seed in [42, 73, 123, 666, 1337]:
            print('Seed:', seed)
            for n_samples in [1000, 5000, 10000]:
                print('Nr of Samples:', n_samples)
                evaluate_synth_data(args.data_file, args.dataset, n_samples, seed)
    else:
        evaluate_synth_data(args.data_file, args.dataset, 1000, 42)


if __name__ == '__main__':
    main()
