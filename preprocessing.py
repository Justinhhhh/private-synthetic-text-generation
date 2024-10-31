import json
import pandas as pd
import os
import datasets
import re


def clean_text(s):
    # Method to clean the messy texts from the spam e-mail dataset
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
    # instructions for models to generate drug reviews
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
    # instructions for models to generate reddit posts
    example['src'] = f"write a post to the {example['label'].replace('self.', '')} community: ".lower()
    example['trg'] = example['text']
    example['label'] = ['self.Anxiety', 'self.bipolar', 'self.depression', 'self.offmychest', 'self.SuicideWatch'
                        ].index(example['label'])
    return example


def drugs_prompt(example):
    # instructions for models to generate drug reviews
    example['src'] = f"write a {['negative', 'positive'][example['label']]} drug review: "
    example['trg'] = example['text'].lower()
    return example


def process_swmh(file_name, output_name):
    data = pd.read_csv(os.path.join('data', 'swmh', file_name), usecols=['text', 'label'])
    ds = datasets.Dataset.from_pandas(data).map(swmh_prompt).select_columns(['src', 'trg', 'label'])
    with open(os.path.join('data', 'swmh', output_name), 'w', encoding='utf-8') as f:
        for example in ds:
            f.write(json.dumps(example) + '\n')
    os.remove(os.path.join('data', 'swmh', file_name))


def main():
    # preprocess thumbs up, we balance all splits

    print("Preprocessing Thumbs Up...")
    thumbs_up = datasets.load_dataset("recmeapp/thumbs-up")

    for key in thumbs_up.keys():
        balanced_df = thumbs_up[key].to_pandas()
        min_class = min(balanced_df.label.value_counts())
        balanced_df = balanced_df.groupby('label').apply(lambda x: x.sample(min_class)).reset_index(drop=True)
        balanced_df = datasets.Dataset.from_pandas(balanced_df)
        balanced_df = balanced_df.map(thumbs_up_prompt).select_columns(['src', 'trg', 'label'])
        with open(os.path.join('data', 'thumbs-up', key + '.jsonl'), 'w', encoding='utf-8') as f:
            for example in balanced_df:
                f.write(json.dumps(example) + '\n')

    # Preprocess WebMD, we balance the dataset and split into train and test
    print("Preprocessing WebMD...")
    webmd_dir = os.path.join('data', 'webmd')
    os.makedirs(webmd_dir, exist_ok=True)
    try:
        ds = datasets.Dataset.from_csv(os.path.join(webmd_dir, 'webmd.csv'))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\nFile not found: 'data/webmd/webmd.csv'. Please make sure the file exists in the 'data' directory.\n"
            "If you don't have the file, you can download it here: \n"
            "https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset"
        )

    ds = ds.filter(lambda x: x['Satisfaction'] <= 5).rename_column('Satisfaction', 'label')
    ds = ds.map(webmd_prompt).select_columns(['src', 'trg', 'label'])

    df = pd.DataFrame(ds)
    min_class = min(df.label.value_counts())
    balanced_df = df.groupby('label').apply(lambda x: x.sample(min_class)).reset_index(drop=True)
    ds = datasets.Dataset.from_pandas(balanced_df).class_encode_column('label')

    # split into train and test
    split_ds = ds.train_test_split(test_size=0.2, shuffle=True, seed=1337, stratify_by_column='label')

    for key in split_ds.keys():
        # write to jsonl
        with open(os.path.join(webmd_dir, key + '.jsonl'), 'w', encoding='utf-8') as f:
            for example in split_ds[key]:
                f.write(json.dumps(example) + '\n')

    # spam detection
    print("Preprocessing SPAM...")
    # move file from kaggle to folder created below
    spam_dir = os.path.join('data', 'spam')
    os.makedirs(spam_dir, exist_ok=True)
    dfs = []

    # manually deleted Enron and phishing_email, we automated it here
    # concatenate the csv from kaggle
    try:
        relevant_files = [file for file in os.listdir(spam_dir) if file.endswith('.csv') and
                          file not in ['Enron.csv', 'phishing_email.csv']]
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\nFile not found: Please make sure the csv files exists in the 'data/spam' directory.\n"
            "If you don't have the files, you can download them here: \n"
            "https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset"
        )
    for file in relevant_files:
        df = pd.read_csv(os.path.join(spam_dir, file), usecols=['subject', 'body', 'label'])
        dfs.append(df)

    # catch empty cells
    df = pd.concat(dfs, ignore_index=True)
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')

    ds = datasets.Dataset.from_pandas(df).map(phishing_prompt).select_columns(['src', 'trg', 'label'])
    ds = ds.map(lambda example: {**example, 'trg': clean_text(example['trg'])})
    ds = ds.class_encode_column('label')
    split_ds = ds.train_test_split(test_size=0.2, shuffle=True, seed=1337, stratify_by_column='label')

    for key in split_ds.keys():
        # write to jsonl
        with open(os.path.join('data', 'phishing', key + '.jsonl'), 'w', encoding='utf-8') as f:
            for example in split_ds[key]:
                f.write(json.dumps(example) + '\n')

    os.rename(spam_dir, os.path.join('data', 'can-be-deleted'), )

    # SWMH
    print("Preprocessing SWMH...")
    os.makedirs(os.path.join('data', 'swmh'), exist_ok=True)
    try:
        process_swmh('train.csv', 'train.jsonl')
        process_swmh('val.csv', 'validation.jsonl')
        process_swmh('test.csv', 'test.jsonl')
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\nFile not found: Please make sure the csv files exists in the 'data/swmh' directory.\n"
            "If you don't have the files, you can download them here: \n"
            "https://zenodo.org/records/6476179"
        )

    # drugs
    # Unnecessary, as we provide the jsonl files in our github repo
    """
    drugs_ds = datasets.Dataset.from_json('data/drugs/all_drugs_com_reviews.json')
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
    """


if __name__ == "__main__":
    main()
