import argparse
import collections
import json
import math
import os

import opacus
import torch
from torch.serialization import default_restore_location
from transformers import AutoTokenizer
from transformers import set_seed

from data_util.s2s_data_util import load_s2s_data, S2S_dataset
from diffusion_util.resample import create_named_schedule_sampler
from train_util.train_util import TrainLoop
from util import logger
from util.util import (
    create_model_and_diffusion,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CheckpointState = collections.namedtuple("CheckpointState",
                                         ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset'])


def get_arguments():
    parser = argparse.ArgumentParser()

    # console commands

    # out path

    # load pretrain
    parser.add_argument('--pretrain_model_path', type=str, default='GENIE_ckpt-500w',
                        help='using pretraining diffusion')

    # data args
    parser.add_argument('--data_dir', type=str, default='', help='data path')
    parser.add_argument('--dataset', type=str, default='thumbs_up', help='data name')

    # training args
    parser.add_argument('--learning_steps', type=int, default=50000, help='total step')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--lr', type=float, default=1e-05, help='')
    parser.add_argument('--warmup_steps', type=int, default=150, help='')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='')
    parser.add_argument('--log_interval', type=int, default=500, help='')
    parser.add_argument('--save_interval', type=int, default=10000, help='')

    # privacy args
    parser.add_argument('--private', type=bool, default=False)
    parser.add_argument('--epsilon', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.)

    # seed
    parser.add_argument('--seed', type=int, default=101, help='')

    args = parser.parse_args()

    # add set default values
    args.model_arch, args.train_type, args.training_mode = 's2s_CAT', 'S2S_Diffusion', 's2s'
    args.model_channels, args.in_channel, args.out_channel = 128, 128, 128
    args.dropout = 0.1
    args.learn_sigma = False
    args.logits_mode = 1
    args.config_name = 'bert-base-uncased'
    args.token_emb_type = 'random'
    args.init_pretrained = False
    args.fix_encoder = False
    args.diffusion_steps = 2000
    args.use_kl = False
    args.noise_schedule = 'sqrt'
    args.predict_xstart = True
    args.sigma_small = False,
    args.rescale_learned_sigmas = True
    args.rescale_timesteps = True
    args.schedule_sampler = 'loss-second-moment'
    args.src_max_len = 16
    args.tgt_max_len = 128
    args.answer_max_len = 128
    args.weight_decay = 0
    args.gradient_clipping = -1.
    args.ema_rate = 0.9999
    args.checkpoint_path = os.path.join('..', 'models', args.dataset,
                                        'eps' + str(args.epsilon) if args.private else 'np', 'genie')

    return args


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)


def main():
    # args setting
    args = get_arguments()

    # out dir set
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    logger.log(f'saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # seed setting
    set_seed(args.seed)
    # gpu settings
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1

    # logger setting
    log_path = os.path.join(args.checkpoint_path, 'log.txt')
    logger.configure(dir=log_path)

    # check paths
    args.data_dir = os.path.join('..', 'data', args.dataset)
    # load data
    tokenizer = AutoTokenizer.from_pretrained(args.config_name)
    args.vocab_size = tokenizer.vocab_size
    train_data = load_s2s_data(
        args,
        tokenizer=tokenizer,
    )
    args.lr_anneal_steps = args.learning_steps

    # load model
    model, diffusion = create_model_and_diffusion(
        args
    )
    if args.pretrain_model_path is not None:
        print("load model ckpt at :", args.pretrain_model_path)
        saved_state = load_states_from_checkpoint(args.pretrain_model_path)
        model.load_state_dict(saved_state.model_dict, strict=False)
    model.to(args.device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # training
    train_sample = torch.utils.data.SequentialSampler(train_data)
    data_loader = torch.utils.data.DataLoader(train_data, sampler=train_sample, batch_size=args.batch_size,
                                              drop_last=False, num_workers=0,
                                              collate_fn=S2S_dataset.get_collate_fn())

    optimizer = torch.optim.AdamW(list(model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # make private
    if args.private:
        privacy_engine = opacus.PrivacyEngine()
        args.delta = 1 / (len(train_data) * 10)
        args.privacy_engine = privacy_engine
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            epochs=math.ceil(args.learning_steps / len(data_loader)),
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            max_grad_norm=args.max_grad_norm,
        )

    logger.log("training Diffusion LM model...")
    TrainLoop(
        # Training Core
        model=model,
        diffusion=diffusion,
        data=train_data,
        schedule_sampler=schedule_sampler,
        optimizer=optimizer,
        train_loader=data_loader,
        # args
        args=args
    ).run_loop()


if __name__ == "__main__":
    main()
