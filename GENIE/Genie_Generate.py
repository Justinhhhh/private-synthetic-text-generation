import argparse
import collections
import json
import logging
import os
import random
from functools import partial

import numpy as np
import pandas as pd
# from transformers import set_seed
import torch
from torch.serialization import default_restore_location
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)

from data_util.s2s_data_util import S2S_dataset
from util.util import (
    create_model_and_diffusion,
)

logging.basicConfig(level=logging.INFO)


def get_arguments():
    parser = argparse.ArgumentParser()

    # out path
    parser.add_argument('--model_path', type=str, default='',
                        help='model path')
    parser.add_argument('--dataset', type=str, default='', help='path to sample data')
    parser.add_argument('--num_samples', type=int, default=1, help='sample query')
    parser.add_argument('--interval_step', type=int, default=1, help='inference t interval step')
    parser.add_argument('--diffusion_steps', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    return args


CheckpointState = collections.namedtuple("CheckpointState",
                                         ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset'])


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    return CheckpointState(**state_dict)


'''
rounding
'''


def denoised_fn_round(args, model, text_emb, t):
    if args.model_arch == '1d-unet':
        text_emb = text_emb.permute(0, 2, 1)

    down_proj_emb = model.weight
    old_shape = text_emb.shape
    old_device = text_emb.device

    def get_efficient_knn(down_proj_emb, text_emb, dist='l2'):
        if dist == 'l2':
            emb_norm = (down_proj_emb ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            # print(emb_norm.shape, arr_norm.shape)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(down_proj_emb,
                                                                        text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            dist = torch.clamp(dist, 0.0, np.inf)

        topk_out = torch.topk(-dist, k=1, dim=0)
        return topk_out.values, topk_out.indices

    def get_knn(down_proj_emb, text_emb, dist='l2'):
        if dist == 'l2':
            adjacency = down_proj_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
                down_proj_emb.size(0), -1, -1)
            adjacency = -torch.norm(adjacency, dim=-1)
        topk_out = torch.topk(adjacency, k=1, dim=0)
        return topk_out.values, topk_out.indices

    dist = 'l2'
    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb

    val, indices = get_efficient_knn(down_proj_emb,
                                     text_emb.to(down_proj_emb.device), dist=dist)
    rounded_tokens = indices[0]
    new_embeds = model(rounded_tokens).view(old_shape).to(old_device)
    if args.model_arch == '1d-unet':
        new_embeds = new_embeds.permute(0, 2, 1)
    return new_embeds


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    # env setting
    args = get_arguments()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # add args from model training
    with open(os.path.join(args.model_path.rsplit('/', 1)[0], 'training_args.json')) as json_file:

        training_args = json.load(json_file)

        for key in training_args.keys():
            if key not in args.__dict__.keys():
                args.__dict__[key] = training_args[key]

    # bert tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.config_name)
    args.vocab_size = tokenizer.vocab_size
    args.answer_max_len = 128
    args.tgt_max_len = 128
    args.sample_data = os.path.join('..', 'data', args.dataset, 'samples.jsonl')

    # define model and diffusion
    model, diffusion = create_model_and_diffusion(
        args
    )
    model.to(args.device)
    model.eval()
    # load trained model
    model_saved_state = load_states_from_checkpoint(args.model_path)
    # remove _module if training was done with opacus
    """if args.private:
        new_model_dict = {key.replace('_module.', ''): value
                          for key, value in model_saved_state.model_dict.items()}

        new_model_dict["passage_encoder.embeddings.position_ids"] = new_model_dict["position_ids"]
        model.load_state_dict(new_model_dict)
    else:
    
    model_saved_state.model_dict["passage_encoder.embeddings.position_ids"] = model_saved_state.model_dict[
        "position_ids"]
    """
    model.load_state_dict(model_saved_state.model_dict)

    sample_fn = (
        diffusion.p_sample_loop
    )

    emb_model = model.word_embedding

    sample_data = pd.read_json(args.sample_data, lines=True)
    sample_data['trg'] = sample_data['trg'].astype(str)

    test_dataset = S2S_dataset(sample_data.src.values, [""] * len(sample_data), tokenizer,
                               src_maxlength=args.src_max_len, tgt_maxlength=args.tgt_max_len,
                               labels=sample_data.label.values.tolist())
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False,
                                 num_workers=0, collate_fn=S2S_dataset.get_collate_fn())

    out_path = os.path.join(os.path.split(args.model_path)[0].replace('models', 'results'), "samples.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for index, batch in enumerate(tqdm(test_dataloader)):
        input_shape = (batch['src_input_ids'].shape[0], args.tgt_max_len, args.in_channel)
        src_input_ids = batch['src_input_ids']
        src_attention_mask = batch['src_attention_mask']
        model_kwargs = {'src_input_ids': src_input_ids, 'src_attention_mask': src_attention_mask}
        orig_label = batch['labels']

        sample = sample_fn(
            model,
            input_shape,
            clip_denoised=False,
            denoised_fn=partial(denoised_fn_round, args, emb_model.cuda()),
            model_kwargs=model_kwargs,
            top_p=-1.0,
            interval_step=args.interval_step,
        )

        logits = model.get_logits(sample)
        cands = torch.topk(logits, k=1, dim=-1)

        with open(out_path, 'a') as fout:
            fout.write(json.dumps({"src": tokenizer.decode(src_input_ids.squeeze(), skip_special_tokens=True),
                                   "trg": tokenizer.decode(cands.indices.squeeze(), skip_special_tokens=True),
                                   "label": orig_label[0]
                                   }) + '\n')


if __name__ == "__main__":
    main()
