import numpy as np
import torch
import json
from tqdm import tqdm
import copy
import os
from functools import partial
import functools
import blobfile as bf
import math

from model_utils import create_model_and_diffusion
from args_utils import create_argparser, model_and_diffusion_defaults
from tokenizer_utils import create_tokenizer
from torch.utils.data import DataLoader, Dataset
from dataloader_utils import TextDataset_translation
from trainer import log_loss_dict
from src.utils import logger

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from src.modeling.diffusion.resample import LossAwareSampler, UniformSampler
from src.modeling.diffusion.gaussian_diffusion import ModelMeanType
from src.modeling.diffusion.gaussian_diffusion import _extract_into_tensor
from src.modeling.diffusion.nn import mean_flat
from src.modeling.diffusion.nn import update_ema
import sys

# Modifed miniconda3/envs/trusthlt/lib/python3.11/site-packages/opacus/grad_sample/grad_sample_module.py (line 286, 390)

with open("args_config.json", 'r') as f:
    args = json.load(f)

dataset = sys.argv[1]
model_path = sys.argv[2]
num_samples = int(sys.argv[3])
num_steps = 100000
batch_size = 128
seq_len = int(sys.argv[4])

# privacy engine hyperparameters
MAX_GRAD_NORM = 1
EPOCHS = math.ceil(num_steps * batch_size / num_samples)
EPSILON = int(sys.argv[5]) # Need to change
DELTA = 1 / (num_samples*10)

args["warmup"] = 2000
args["batch_size"] = batch_size
args["lr_anneal_steps"] = num_steps
args["train_txt_path"] = f"data/{dataset}/train"
#args["checkpoint_path"] = f"ckpts/e{EPSILON}/{dataset}/{num_steps}/"
args["save_interval"] = 20000
logger.configure(dir=os.path.join(os.getcwd(), f'logs/{dataset}'))
print(args["train_txt_path"])
print(args["checkpoint_path"])

tokenizer = create_tokenizer(return_pretokenized=args['use_pretrained_tokenizer'],
                             path=f"data/{args['dataset']}/",
                             tokenizer_type='byte-level',
                             tokenizer_ckpt=args['pretrained_tokenizer'])

n = 120000
#model_path = f"ckpts/{dataset}_ckpts/ema_0.9999_{n}.pt"
#diff_path = f"ckpts/{dataset}_ckpts/alpha_cumprod_step_{n}.npy"

args['load_ckpt'] = model_path # Using saved checkpoint
args["checkpoint_path"] = f"ckpts/e{EPSILON}/{dataset}_cc/{num_steps}/"
#args['load_ckpt'] = None # Using new model

args['vocab_size'] = tokenizer.vocab_size
args['sequence_len_src'] = seq_len
args['sequence_len'] = seq_len
def args_to_dict(args, keys):
    return {k: args[k] for k in keys}

model, diffusion = create_model_and_diffusion(
    pad_tok_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else tokenizer.get_vocab()['<pad>'],
    resume_checkpoint=args['checkpoint_path'], 
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)
#diffusion._load_time_schedule(diff_path)

for name, param in model.named_parameters():
    if "embed_positions" in name:
        param.requires_grad = False

def get_dataloader(tokenizer, data_path, batch_size, max_seq_len, max_seq_len_src, args):

    dataset = TextDataset_translation(tokenizer=tokenizer, data_path=data_path, source=args['src'], target=args['tgt'],
                                        shard=1,
                                        num_shards=1)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # 20,
        #drop_last=True,
        shuffle='train' in data_path,
        num_workers=0,
        collate_fn=partial(TextDataset_translation.collate_pad, 
                           args=args,
                           cutoff=max_seq_len, 
                           cutoff_src=max_seq_len_src,
                           padding_token=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else tokenizer.get_vocab()['<pad>']),
    )
    return dataloader

train_dataloader = get_dataloader(
    tokenizer=tokenizer,
    args=args,
    data_path=args['train_txt_path'],
    batch_size=args['batch_size'],
    max_seq_len=args['sequence_len'],
    max_seq_len_src=args['sequence_len_src'],
)

privacy_engine = PrivacyEngine()
optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"])

model.train()
model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_dataloader,
    target_delta=DELTA,
    target_epsilon=EPSILON,
    epochs=EPOCHS,
    max_grad_norm=MAX_GRAD_NORM,
)

ema_rate = (
    [args["ema_rate"]]
    if isinstance(args["ema_rate"], float)
    else [float(x) for x in args["ema_rate"].split(",")]
)

ema_params = [
    copy.deepcopy(list(model._module.parameters())) for _ in range(len(ema_rate))
]
#ema_params = torch.tensor(ema_params).to(device)

def anneal_lr(args, optimizer, step):
    if "lr_anneal_steps" not in args:
        return
    
    if "warmup" in args and args["warmup"] > 0:
        warmup_frac = step / args["warmup"]
        frac_done = (step - args["warmup"]) / (args["lr_anneal_steps"]  - args["warmup"])
        
        lr = args["lr"] * min(1, warmup_frac) * min(1-frac_done, 1)

    else:
        frac_done = step / args["lr_anneal_steps"]
        lr = args["lr"] * (1 - frac_done)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def log_grad_norm(model):
    sqsum = 0.0
    for p in model._module.parameters():
        if p.grad is not None:
            sqsum += (p.grad ** 2).sum().item()
    logger.logkv_mean("grad_norm", np.sqrt(sqsum))

def save_checkpoint(rate, params, step):
    state_dict = model._module.state_dict()
    master_params = list(model._module.parameters())
    for i, (name, _value) in enumerate(model._module.named_parameters()):
        assert name in state_dict
        state_dict[name] = master_params[i]
    logger.log(f"saving model {rate}...")
    if not rate:
        filename = f"model{(step):06d}.pt"
    else:
        filename = f"ema_{rate}_{(step):06d}.pt"
    #print('writing to', bf.join(get_blob_logdir(), filename))
    print('writing to', bf.join(args["checkpoint_path"], filename))
    with bf.BlobFile(bf.join(args["checkpoint_path"], filename), "wb") as f: # DEBUG **
        torch.save(state_dict, f)

schedule_sampler = UniformSampler(diffusion)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
logger.log("training...")
step = 0
with BatchMemoryManager(
    data_loader=train_dataloader,
    max_physical_batch_size=64,
    optimizer=optimizer
) as memory_safe_data_loader:
    for epoch in range(EPOCHS):
        for i, (batch, cond) in enumerate(memory_safe_data_loader):
            if step%500 == 0:
                print(f"epoch:{epoch} , step:{step}")
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            t, weights = schedule_sampler.sample(batch['input_ids'].shape[0], device)
            
            compute_losses = functools.partial(
                diffusion.training_losses,
                model,
                step,
                t,
                model_kwargs=inputs,
            )
            
            losses = compute_losses()
            loss = (losses["loss"] * weights).mean()
            log_loss_dict(diffusion, t, {k: v * weights for k, v in losses.items()})
            loss.backward()
            log_grad_norm(model)
            anneal_lr(args, optimizer, 0)
            optimizer.step()
            for rate, params in zip(ema_rate, ema_params):
                params = [p.to(device) for p in params]
                update_ema(params, list(model._module.parameters()), rate=rate)
            if step % args["log_interval"] == 0:
                logger.dumpkvs()
            if step % args["save_interval"] == 0:
                save_checkpoint(0, model._module.parameters(), step)
                for rate, params in zip(ema_rate, ema_params):
                    save_checkpoint(rate, params, step)
                np.save(os.path.join(args["checkpoint_path"], f'alpha_cumprod_step_{step}.npy'), diffusion.alphas_cumprod)
                np.save(os.path.join(args["checkpoint_path"], f'loss_step_{step}.npy'), diffusion._loss_history)
                np.save(os.path.join(args["checkpoint_path"], f'loss_count_{step}.npy'), diffusion._loss_history_count)
            step += 1
                    
save_checkpoint(0, model._module.parameters(), step)
for rate, params in zip(ema_rate, ema_params):
    save_checkpoint(rate, params, step)
np.save(os.path.join(args["checkpoint_path"], f'alpha_cumprod_step_{step}.npy'), diffusion.alphas_cumprod)
np.save(os.path.join(args["checkpoint_path"], f'loss_step_{step}.npy'), diffusion._loss_history)
np.save(os.path.join(args["checkpoint_path"], f'loss_count_{step}.npy'), diffusion._loss_history_count)

path = os.path.join(args["checkpoint_path"], 'training_args.json') 
with open(path, 'w') as fp:
    json.dump(args, fp)