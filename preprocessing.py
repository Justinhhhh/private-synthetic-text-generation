import json
import pandas as pd
import os
import datasets
import re


def clean_text(s):
    a = re.sub("[^a-zA-Z\d\s:/=.]", "", s)
    a = a.replace("\n", " ")
    a = a.replace("\t", " ")
    return a


def thumbs_up_prompt(example):
    # instructions for models to generate synth. app reviews based on thumbs-up
    example['src'] = f"write a {['mild', 'notable', 'concerning', 'serious', 'hot'][example['label']]} app review: "
    example['trg'] = example['review'].lower() if example['review'] else ''
    return example


def webmd_prompt(example):
    # instructions for models to generate medicine reviews
    example['src'] = \
        f"write a {['terrible', 'poor', 'neutral', 'good', 'great'][example['label'] - 1]} medicine review: "
    example['trg'] = example['Reviews'].lower() if example['Reviews'] else ''
    return example


def phishing_prompt(example):
    # instructions for models to generate mails
    example['src'] = f"write a {['non-spam', 'spam'][example['label']]} e-mail: "
    example['trg'] = example['subject'].lower() + ': ' + example['body'].lower()
    return example


def swmh_prompt(example):
    example['src'] = f"write a post to the {example['label'].replace('self.', '')} community: ".lower()
    example['trg'] = example['text']
    example['label'] = ['self.Anxiety', 'self.bipolar', 'self.depression', 'self.offmychest', 'self.SuicideWatch'
                        ].index(example['label'])
    return example


def drugs_prompt(example):
    example['src'] = f"write a {['negative', 'positive'][example['label']]} drug review: "
    example['trg'] = example['text'].lower()
    return example


def process_swmh(file_name, output_name):
    data = pd.read_csv(os.path.join('data', 'swmh', file_name), usecols=['text', 'label'])
    ds = datasets.Dataset.from_pandas(data).map(swmh_prompt).select_columns(['src', 'trg', 'label'])
    with open(os.path.join('data', 'swmh', output_name), 'w', encoding='utf-8') as f:
        for example in ds:
            f.write(json.dumps(example) + '\n')


def map_prompt_select_columns(dataset):
    return dataset.map(thumbs_up_prompt).select_columns(['src', 'trg', 'label'])


def main():
    # balance thumbs up data set for synthetic text generation
    balanced = {}
    thumbs_up = datasets.load_dataset("recmeapp/thumbs-up")

    for key in thumbs_up.keys():
        df = pd.DataFrame(thumbs_up[key])
        min_class = min(df.label.value_counts())
        balanced_df = df.groupby('label').apply(lambda x: x.sample(min_class)).reset_index(drop=True)
        balanced_df = datasets.Dataset.from_pandas(balanced_df)
        balanced[key] = map_prompt_select_columns(balanced_df)

    datasets.DatasetDict(balanced).save_to_disk('data/thumbs-up-balanced')

    for key in balanced.keys():
        # write to jsonl
        with open('data/thumbs-up/' + key + '.jsonl', 'w', encoding='utf-8') as f:
            for example in balanced[key]:
                f.write(json.dumps(example) + '\n')

    # balance and save webmd
    ds = datasets.Dataset.from_csv('data/webmd.csv')
    ds = ds.filter(lambda x: x['Satisfaction'] <= 5).rename_column('Satisfaction', 'label')
    ds = ds.map(webmd_prompt).select_columns(['src', 'trg', 'label'])

    df = pd.DataFrame(ds)
    min_class = min(df.label.value_counts())
    balanced_df = df.groupby('label').apply(lambda x: x.sample(min_class)).reset_index(drop=True)
    ds = datasets.Dataset.from_pandas(balanced_df).class_encode_column('label')

    # split into train and test
    split_ds = ds.train_test_split(test_size=0.2, shuffle=True, seed=1337, stratify_by_column='label')
    datasets.DatasetDict(split_ds).save_to_disk('data/webmd')

    for key in split_ds.keys():
        # write to jsonl
        with open('data/webmd/' + key + '.jsonl', 'w', encoding='utf-8') as f:
            for example in split_ds[key]:
                f.write(json.dumps(example) + '\n')

    # spam detection
    dfs = []

    for file in [file for file in os.listdir('data/spam') if file.endswith('.csv')]:
        df = pd.read_csv(os.path.join('data/spam', file), usecols=['subject', 'body', 'label'])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')

    ds = datasets.Dataset.from_pandas(df).map(phishing_prompt).select_columns(['src', 'trg', 'label'])
    ds = ds.class_encode_column('label')
    split_ds = ds.train_test_split(test_size=0.2, shuffle=True, seed=1337, stratify_by_column='label')

    datasets.DatasetDict(split_ds).save_to_disk('data/phishing')
    for key in split_ds.keys():
        # write to jsonl
        with open('data/phishing/' + key + '.jsonl', 'w', encoding='utf-8') as f:
            for example in split_ds[key]:
                f.write(json.dumps(example) + '\n')

    for key in ['train', 'test']:
        cleaned = []

        with open('data/phishing_uncleaned/' + key + '.jsonl', 'r') as f_reader:
            for row in f_reader:
                content = json.loads(row)
                content['trg'] = clean_text(content['trg'])
                cleaned.append(content)
        os.makedirs(os.path.dirname('data/phishing/' + key + '.jsonl'), exist_ok=True)
        with open('data/phishing/' + key + '.jsonl', 'w') as f_writer:
            for example in cleaned:
                f_writer.write(json.dumps(example) + '\n')

    # SWMH
    process_swmh('train.csv', 'train.jsonl')
    process_swmh('val.csv', 'validation.jsonl')
    process_swmh('test.csv', 'test.jsonl')

    # drugs
    drugs_ds = datasets.Dataset.from_json('data/drugs/final_results.json')
    # shuffle. only 1 author per review, add label
    drugs_ds = datasets.Dataset.from_pandas(drugs_ds.to_pandas().sample(frac=1, random_state=42).drop_duplicates(
        subset='user_name', keep='first').reset_index(drop=True))

    drugs_ds = drugs_ds.add_column(
        'label', [1 if x > 5 else 0 for x in drugs_ds['rating']])

    # split into train, dev, split
    train_test = drugs_ds.train_test_split(test_size=0.2)
    drugs_train = train_test['train'].map(drugs_prompt).select_columns(['src', 'trg', 'label'])
    temp_test = train_test['test']

    dev_test = temp_test.train_test_split(test_size=0.5)
    drugs_dev = dev_test['train'].map(drugs_prompt).select_columns(['src', 'trg', 'label'])
    drugs_test = dev_test['test'].map(drugs_prompt).select_columns(['src', 'trg', 'label'])

    with open('data/drugs/train.jsonl', 'w') as f_writer:
        for example in drugs_train:
            f_writer.write(json.dumps(example) + '\n')

    with open('data/drugs/val.jsonl', 'w') as f_writer:
        for example in drugs_dev:
            f_writer.write(json.dumps(example) + '\n')

    with open('data/drugs/test.jsonl', 'w') as f_writer:
        for example in drugs_test:
            f_writer.write(json.dumps(example) + '\n')


if __name__ == "__main__":
    main()
