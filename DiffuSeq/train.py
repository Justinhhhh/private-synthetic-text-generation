"""
Train a diffusion model on images.
"""

import argparse
import json
import math
import os

import opacus
import torch
from transformers import set_seed

from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_model_emb,
    load_tokenizer
)
from diffuseq.step_sample import create_named_schedule_sampler
from diffuseq.text_datasets import load_data_text
from diffuseq.utils import logger, dist_util
from train_util import TrainLoop

os.environ["WANDB_MODE"] = "offline"


def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    parser.add_argument('--private', type=bool, default=False, help='run fine-tuning with DP')
    parser.add_argument('--epsilon', type=int, default=1, help='if dp, how large should epsilon be?')

    add_dict_to_argparser(parser, defaults)  # update latest args according to argparse

    return parser


def main():
    args = create_argparser().parse_args()
    # update args
    args.save_interval = 10000
    if args.private:
        args.checkpoint_path = os.path.join('..', 'models', args.dataset, 'eps' + str(args.epsilon), 'diffuseq')
    else:
        args.checkpoint_path = os.path.join('..', 'models', args.dataset, 'np', 'diffuseq')

    args.config_name = 'bert-base-uncased'
    args.log_interval = 500

    set_seed(args.seed)
    logger.configure()

    # load tokenizer and create model embeds
    tokenizer = load_tokenizer(args)

    if args.private:
        model_weight, tokenizer = load_model_emb(args, tokenizer)
    else:
        model_weight, tokenizer = load_model_emb(args, tokenizer)

    # load train data
    data = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args=args,
        loaded_vocab=tokenizer,
        model_emb=model_weight  # use model's weights as init
    )

    # create model and diffusion mechanism
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    if args.private:
        # load pretrained model for training
        print("loading qqp model for private training")
        model_dict = dist_util.load_state_dict("models/pretrained/pretrained_weights.pt")

        model.load_state_dict(model_dict)

    model.cuda()

    # load optimizer and create lr schedule
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # save training args for later inference
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    logger.log("### Training...")

    # make private
    if args.private:
        privacy_engine = opacus.PrivacyEngine()
        model, opt, data = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=opt,
            data_loader=data,
            target_epsilon=args.epsilon,
            target_delta=(len(data.dataset) * 10) ** -1,
            epochs=math.ceil(args.learning_steps / len(data)),
            max_grad_norm=1.0
        )

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=None,
        eval_interval=None,
        opt=opt,
        private=args.private,
        args=args
    ).run_loop()


if __name__ == "__main__":
    main()
