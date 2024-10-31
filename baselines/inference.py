import json
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import datasets
import os
from tqdm.auto import tqdm


def inference(path_to_baseline, data_set):
    def tokenize(example):
        return tokenizer(example['src'], padding='max_length', max_length=16, add_special_tokens=True)

    model_check = os.path.normpath(path_to_baseline).split(os.sep)
    # load model & tokenizer
    if 'bart' in model_check:
        conditional = True
        model = AutoModelForSeq2SeqLM.from_pretrained(path_to_baseline)
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large', padding_side="left")
    else:
        conditional = False
        model = AutoModelForCausalLM.from_pretrained(path_to_baseline)
        if 'bloom' in model_check:
            tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-560m', padding_side="left")
            tokenizer.pad_token = tokenizer.eos_token

        elif 'phi' in model_check:
            tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-1_5', padding_side="left")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            NotImplementedError("Model not detected in file path ¯\_(ツ)_/¯")

    model.to('cuda')
    model.eval()

    # prep data
    data_path = os.path.join('..', 'data', data_set, 'samples.jsonl')
    data_set = datasets.Dataset.from_json(data_path)

    eval_loader = torch.utils.data.DataLoader(
        data_set.map(tokenize, remove_columns=data_set.column_names).with_format('torch'),
        batch_size=8,
        shuffle=False,
        num_workers=0)

    gen = []
    for batch in tqdm(eval_loader, desc="decoding", unit="batch"):
        inputs = {k: v.to('cuda') for k, v in batch.items()}
        outputs = model.generate(**inputs, max_new_tokens=256, no_repeat_ngram_size=3, early_stopping=True,
                                 do_sample=True)
        if conditional:
            gen.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        else:
            gen.extend(tokenizer.batch_decode(outputs[:, 16:], skip_special_tokens=True))

    results = data_set.to_dict()
    results['trg'] = gen

    out_path = os.path.join(os.path.split(path_to_baseline)[0].replace('models', 'results'), "samples.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(os.path.join(os.path.split(path_to_baseline.replace('models', 'results'))[0], 'samples.jsonl'), 'w') as f:
        for i in range(len(gen)):
            row = {key: values[i] for key, values in results.items()}
            f.write(json.dumps(row) + '\n')




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Name of the model')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    args = parser.parse_args()
    inference(args.model_path, args.dataset)


if __name__ == '__main__':
    main()
