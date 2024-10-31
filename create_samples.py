import json
import os
import datasets


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


def map_prompt_select_columns(dataset):
    return dataset.map(thumbs_up_prompt).select_columns(['src', 'trg', 'label'])


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


def main():
    for sub_folder in os.listdir('data'):
        if sub_folder == 'phishing':
            ds = datasets.Dataset.from_dict(
                {'subject': [''] * 1000, 'body': [''] * 1000, 'label': [0, 1] * 500}).map(phishing_prompt)
            ds = ds.map(lambda x: {'trg': ''}).select_columns(['src', 'trg', 'label'])
            with open(os.path.join('data', 'phishing', 'samples.jsonl'), 'w', encoding='utf-8') as f:
                for example in ds:
                    f.write(json.dumps(example) + '\n')
        if sub_folder == 'thumbs-up':
            ds = datasets.Dataset.from_dict({'review': [''] * 1000, 'label': [0, 1, 2, 3, 4] * 200}).map(
                thumbs_up_prompt)
            ds = ds.select_columns(['src', 'trg', 'label'])
            with open(os.path.join('data', 'thumbs-up', 'samples.jsonl'), 'w', encoding='utf-8') as f:
                for example in ds:
                    f.write(json.dumps(example) + '\n')
        if sub_folder == 'webmd':
            ds = datasets.Dataset.from_dict({'Reviews': [''] * 1000, 'label': [0, 1, 2, 3, 4] * 200}).map(webmd_prompt)
            ds = ds.select_columns(['src', 'trg', 'label'])
            with open(os.path.join('data', 'webmd', 'samples.jsonl'), 'w', encoding='utf-8') as f:
                for example in ds:
                    f.write(json.dumps(example) + '\n')
        if sub_folder == 'swmh':
            ds = datasets.Dataset.from_dict({'text': [''] * 1000,
                                             'label': ['self.Anxiety', 'self.bipolar',
                                                       'self.depression', 'self.offmychest',
                                                       'self.SuicideWatch'] * 200}).map(swmh_prompt)
            ds = ds.select_columns(['src', 'trg', 'label'])
            with open(os.path.join('data', 'swmh', 'samples.jsonl'), 'w', encoding='utf-8') as f:
                for example in ds:
                    f.write(json.dumps(example) + '\n')
        if sub_folder == 'drugs':
            ds = datasets.Dataset.from_dict({'text': [''] * 10000,
                                             'label': [0, 1] * 5000}).map(drugs_prompt)
            ds = ds.select_columns(['src', 'trg', 'label'])
            with open(os.path.join('data', 'drugs', 'samples.jsonl'), 'w', encoding='utf-8') as f:
                for example in ds:
                    f.write(json.dumps(example) + '\n')


if __name__ == "__main__":
    main()
