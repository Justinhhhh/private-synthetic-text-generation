import os

import jsonlines
import torch
from torch.utils.data.dataset import Dataset


def load_s2s_data(args, tokenizer):

    src = []
    trg = []

    with jsonlines.open(os.path.join(args.data_dir, 'train.jsonl'), 'r') as reader:
        for line in reader:
            src.append(line['src'])
            trg.append(line['trg'])

    # tokenization
    dataset = S2S_dataset(src, trg, tokenizer, src_maxlength=args.src_max_len,
                          tgt_maxlength=args.tgt_max_len)

    print("total dataset len :", len(dataset))

    return dataset


class QG_dataset_Diff(Dataset):
    def __init__(self, src, tgt, tokenizer, src_maxlength=144, answer_maxlength=20, tgt_maxlength=32):
        self.src = src
        self.tgt = tgt
        self.tokenizer = tokenizer
        self.src_maxlength = src_maxlength
        self.tgt_maxlength = tgt_maxlength
        self.ans_maxlength = answer_maxlength

    def __getitem__(self, index):
        src_example = self.src[index]
        tgt_example = self.tgt[index]

        answer = src_example.split('[SEP]')[0].strip()
        passage = src_example.split('[SEP]')[1].strip()

        src_input_ids = self.tokenizer.encode(passage, add_special_tokens=True,
                                              max_length=self.src_maxlength, truncation=True,
                                              padding='max_length', return_tensors='pt')
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=True,
                                           max_length=self.ans_maxlength, truncation=True,
                                           padding='max_length', return_tensors='pt')
        tgt_input_ids = self.tokenizer.encode(tgt_example, add_special_tokens=True,
                                              max_length=self.tgt_maxlength, truncation=True,
                                              padding='max_length', return_tensors='pt')

        return src_input_ids, answer_ids, tgt_input_ids

    def __len__(self):
        return len(self.src)

    @classmethod
    def get_collate_fn(cls):
        def fn(features):
            src_tensor = torch.cat([feature[0] for feature in features])
            ans_tensor = torch.cat([feature[1] for feature in features])
            tgt_tensor = torch.cat([feature[2] for feature in features])
            return {"src_input_ids": src_tensor, "src_attention_mask": (src_tensor != 0).long(),
                    "answer_ids": ans_tensor, "answer_mask": (ans_tensor != 0).long(),
                    "tgt_input_ids": tgt_tensor, "tgt_attention_mask": (tgt_tensor != 0).long()}

        return fn


class S2S_dataset(Dataset):
    def __init__(self, src, tgt, tokenizer, src_maxlength=144, tgt_maxlength=32, labels=None):
        self.src = src
        self.tgt = tgt
        self.tokenizer = tokenizer
        self.src_maxlength = src_maxlength
        self.tgt_maxlength = tgt_maxlength
        self.labels = labels

    def __getitem__(self, index):
        src_example = self.src[index]
        tgt_example = self.tgt[index]
        if self.labels:
            label_example = self.labels[index]

        src_input_ids = self.tokenizer.encode(src_example, add_special_tokens=True,
                                              max_length=self.src_maxlength, truncation=True,
                                              padding='max_length', return_tensors='pt')
        tgt_input_ids = self.tokenizer.encode(tgt_example, add_special_tokens=True,
                                              max_length=self.tgt_maxlength, truncation=True,
                                              padding='max_length', return_tensors='pt')
        if self.labels:
            return src_input_ids, tgt_input_ids, label_example
        else:
            return src_input_ids, tgt_input_ids

    def __len__(self):
        return len(self.src)

    @classmethod
    def get_collate_fn(cls):
        def fn(features):
            src_tensor = torch.cat([feature[0] for feature in features])
            tgt_tensor = torch.cat([feature[1] for feature in features])
            if len(features[0]) == 2:
                return {"src_input_ids": src_tensor, "src_attention_mask": (src_tensor != 0).long(),
                 "tgt_input_ids": tgt_tensor, "tgt_attention_mask": (tgt_tensor != 0).long()}
            else:
                return {"src_input_ids": src_tensor, "src_attention_mask": (src_tensor != 0).long(),
                        "tgt_input_ids": tgt_tensor, "tgt_attention_mask": (tgt_tensor != 0).long(),
                        "labels": [feature[2] for feature in features]}

        return fn


class S2S_imp_dataset(Dataset):
    def __init__(self, src, tgt, ori_gen, tokenizer, src_maxlength=144, tgt_maxlength=32):
        self.src = src
        self.tgt = tgt
        self.ori_gen = ori_gen
        self.tokenizer = tokenizer
        self.src_maxlength = src_maxlength
        self.tgt_maxlength = tgt_maxlength

    def __getitem__(self, index):
        src_example = self.src[index]
        tgt_example = self.tgt[index]
        ori_gen_example = self.ori_gen[index]

        src_input_ids = self.tokenizer.encode(src_example, add_special_tokens=True,
                                              max_length=self.src_maxlength, truncation=True,
                                              padding='max_length', return_tensors='pt')
        tgt_input_ids = self.tokenizer.encode(tgt_example, add_special_tokens=True,
                                              max_length=self.tgt_maxlength, truncation=True,
                                              padding='max_length', return_tensors='pt')
        ori_gen_ids = self.tokenizer.encode(ori_gen_example, add_special_tokens=True,
                                            max_length=self.tgt_maxlength, truncation=True,
                                            padding='max_length', return_tensors='pt')

        return src_input_ids, tgt_input_ids, ori_gen_ids

    def __len__(self):
        return len(self.src)

    @classmethod
    def get_collate_fn(cls):
        def fn(features):
            src_tensor = torch.cat([feature[0] for feature in features])
            tgt_tensor = torch.cat([feature[1] for feature in features])
            ori_gen_tensor = torch.cat([feature[2] for feature in features])
            return {"src_input_ids": src_tensor, "src_attention_mask": (src_tensor != 0).long(),
                    "tgt_input_ids": tgt_tensor, "tgt_attention_mask": (tgt_tensor != 0).long(),
                    "ori_gen_ids": ori_gen_tensor}

        return fn


if __name__ == "__main__":
    pass
