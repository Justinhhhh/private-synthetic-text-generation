"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import json
import os
import time
from functools import partial

import torch as th
import torch.distributed as dist
from tqdm import tqdm
from transformers import set_seed, AutoTokenizer

from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_tokenizer
)
from diffuseq.rounding import denoised_fn_round
from diffuseq.text_datasets import load_data_text
from diffuseq.utils import dist_util, logger


# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0)
    decode_defaults = dict(split='samples', clamp_step=0, seed2=105, clip_denoised=False, seed=101, step=2000,
                           top_p=-1, pattern='ema')
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


@th.no_grad()
def main():
    args = create_argparser().parse_args()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], 'training_args.json')
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = 1
    training_args['data_dir'] = os.path.join('..', 'data', args.model_path.split('/')[2])
    training_args['dataset'] = args.model_path.split('/')[2]
    args.__dict__.update(training_args)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model_dict = dist_util.load_state_dict(args.model_path)

    model.load_state_dict(model_dict)
    model.eval().requires_grad_(False).to('cuda')

    tokenizer = load_tokenizer(args)
    auto_tokenizer = AutoTokenizer.from_pretrained(args.config_name)
    model_emb = th.nn.Embedding(
        num_embeddings=tokenizer.vocab_size,
        embedding_dim=args.hidden_dim,
        _weight=model.word_embedding.weight.clone().cpu()
    ).eval().requires_grad_(False)

    set_seed(args.seed2)

    ## load data
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb,  # using the same embedding weight with tranining data
        loop=False
    )

    start_t = time.time()

    out_path = os.path.join(os.path.split(args.model_path)[0].replace('models', 'results'), "samples.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # model_emb.to('cuda')

    for batch, cond in tqdm(data_valid):

        input_ids_x = cond.pop('input_ids').to('cuda')
        x_start = model.get_embeds(input_ids_x)
        input_ids_mask = cond.pop('input_mask')
        input_ids_mask_ori = input_ids_mask

        noise = th.randn_like(x_start)
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to('cuda')
        x_noised = th.where(input_ids_mask == 0, x_start, noise)

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps // args.step

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)

        samples = sample_fn(
            model,
            sample_shape,
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_start,
            gap=1,
        )

        # print(samples[0].shape) # samples for each step

        sample = samples[-1]

        # print('decoding for seq2seq', )
        # print(sample.shape)

        logits = model.get_logits(sample)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)

        with open(out_path, 'a') as fout:
            src = auto_tokenizer.decode(input_ids_x.flatten().tolist(), skip_special_tokens=True)
            tgt = auto_tokenizer.decode(cands.indices.flatten().tolist(), skip_special_tokens=True
                                        ).replace(src, '', 1)

            fout.write(json.dumps({"src": src, "trg": tgt, "label": cond['label'].item(), }) + '\n')


    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    main()
