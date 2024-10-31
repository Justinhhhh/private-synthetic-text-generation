import collections
import copy
import math
import os

import numpy as np
import torch
from diffusion_util.resample import LossAwareSampler
from opacus.utils import batch_memory_manager
from torch import nn
from torch.serialization import default_restore_location
from tqdm import tqdm
from transformers import (
    get_linear_schedule_with_warmup,
)
from contextlib import nullcontext
from util import logger

INITIAL_LOG_LOSS_SCALE = 20.0
CheckpointState = collections.namedtuple("CheckpointState",
                                         ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset'])

'''
TrainLoop training class
'''


class TrainLoop:
    def __init__(
            self,
            model,
            diffusion,
            data,
            optimizer,
            train_loader,
            schedule_sampler,
            args
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.args = args
        self.schedule_sampler = schedule_sampler
        # fix ema
        self.ema_rate = ([self.args.ema_rate] if isinstance(self.args.ema_rate, float)
                         else [float(x) for x in self.args.ema_rate.split(",")]
                         )

        self.master_params = list(self.model.parameters())
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.lr_anneal_steps
        )
        self.global_step = 0

        # load last checkpoint
        if self.args.checkpoint_path is not None:
            model_checkpoint_files = []
            ema_checkpoint_files = []
            if os.path.exists(self.args.checkpoint_path):
                for item in os.scandir(self.args.checkpoint_path):
                    if item.is_file():
                        if "model_checkpoint" in item.path:
                            model_checkpoint_files.append(item.path)
                        if "ema" in item.path:
                            ema_checkpoint_files.append(item.path)
                if len(model_checkpoint_files) != 0 and len(ema_checkpoint_files) != 0:
                    model_checkpoint_files.sort(key=lambda f: int(f.split('model_checkpoint-')[1]), reverse=True)
                    logger.info("***** load " + model_checkpoint_files[0] + " *****")
                    ema_checkpoint_files.sort(key=lambda f: int(f.split('checkpoint-')[-1]), reverse=True)
                    logger.info("***** load " + ema_checkpoint_files[0] + " *****")
                    model_saved_state = load_states_from_checkpoint(model_checkpoint_files[0])
                    self.global_step = self._load_saved_state(model_saved_state)
                    self.ema_params = [
                        copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
                    ]
                else:
                    self.ema_params = [
                        copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
                    ]
                    logger.info("***** there are no checkpoint in " + self.args.checkpoint_path + " *****")

        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

    def run_loop(self):
        logger.info("***** Running training *****")
        logger.info("  Max steps = %d", self.args.lr_anneal_steps)
        logger.info("  Instantaneous batch size per GPU = %d", self.args.batch_size)
        self.model.zero_grad()
        self.model.train()

        # memory safe for poisson sampling
        with batch_memory_manager.BatchMemoryManager(
                data_loader=self.train_loader,
                max_physical_batch_size=64,
                optimizer=self.optimizer) if self.args.private else nullcontext() as train_dataloader:

            if not self.args.private:
                train_dataloader = self.train_loader

            pbar = tqdm(total=self.args.learning_steps)

            for epoch in range(math.ceil(self.args.learning_steps / len(train_dataloader))):

                for batch in train_dataloader:

                    # skip zero batches from poisson sampling
                    if type(batch) == list:
                        continue

                    self.model.train()

                    # forward loss
                    self.forward_backward(batch)

                    self._log_grad_norm()

                    if self.args.private:
                        # fix grads for positional embeddings
                        self.model._module.passage_encoder.embeddings.position_embeddings.weight.grad_sample = (
                            self.model._module.passage_encoder.embeddings.position_embeddings.weight.grad_sample.expand(
                                self.model._module.lm_head.weight.grad_sample.shape[0], -1, -1))
                        self.model._module.position_embeddings.weight.grad_sample = (
                            self.model._module.position_embeddings.weight.grad_sample.expand(
                                self.model._module.lm_head.weight.grad_sample.shape[0], -1, -1))

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # lr scheduler
                    self.scheduler.step()
                    self.model.zero_grad()
                    # ema
                    for rate, params in zip(self.ema_rate, self.ema_params):
                        self.update_ema(params, self.master_params, rate=rate)
                    self.global_step += 1
                    pbar.update(1)
                    # self.log_step()

                    if self.global_step % self.args.log_interval == 0:
                        if self.args.private:
                            logger.log("spent epsilon: ", self.args.privacy_engine.get_epsilon(self.args.delta))
                        # logger.dumpkvs()

                    # save
                    if self.global_step % self.args.save_interval == 0:
                        self.save()
            # save after training ended
            self.save()

    def save(self):

        def save_checkpoint():
            model_to_save = get_model_obj(self.model)
            if self.args.private:
                model_state_dict = model_to_save._module.state_dict()
            else:
                model_state_dict = model_to_save.state_dict()

            opt_state_dict = self.optimizer.state_dict()
            sch_state_dict = self.scheduler.state_dict()
            offset = self.global_step
            state = CheckpointState(model_state_dict,
                                    opt_state_dict,
                                    sch_state_dict,
                                    offset,
                                    )

            ckpt_path = os.path.join(self.args.checkpoint_path, str(offset))
            torch.save(state._asdict(), ckpt_path)

        save_checkpoint()

    def forward_backward(self, batch):
        # for s2s
        t, weights = self.schedule_sampler.sample(batch['src_input_ids'].shape[0], self.args.device)
        losses = self.diffusion.training_losses(self.model, batch, t)

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_all_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()
        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )

        loss.backward()

    def forward_only(self, batch):
        with torch.no_grad():
            self.model.zero_grad()
            # for s2s
            t, weights = self.schedule_sampler.sample(batch['src_input_ids'].shape[0], self.args.device)
            losses = self.diffusion.training_losses(self.model, batch, t)

            log_loss_dict(
                self.diffusion, t, {f"eval_{k}": v * weights for k, v in losses.items()}
            )

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            # print(p)
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def log_step(self):
        logger.logkv("step", self.global_step)

    def _anneal_lr(self):
        if not self.args.lr_anneal_steps:
            return
        frac_done = self.global_step / self.args.lr_anneal_steps
        lr = self.args.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _load_saved_state(self, saved_state: CheckpointState):
        self.global_step = saved_state.offset
        logger.info('Loading checkpoint @ step=%s', self.global_step)

        logger.info('Loading saved model state ...')
        self.model.load_state_dict(saved_state.model_dict)  # set strict=False if you use extra projection
        self.optimizer.load_state_dict(saved_state.optimizer_dict)
        self.scheduler.load_state_dict(saved_state.scheduler_dict)
        self.master_params = list(self.model.parameters())
        return self.global_step

    def update_ema(self, target_params, source_params, rate=0.99):
        """
        Update target parameters to be closer to those of source parameters using
        an exponential moving average.

        :param target_params: the target parameter sequence.
        :param source_params: the source parameter sequence.
        :param rate: the EMA rate (closer to 1 means slower).
        """
        for targ, src in zip(target_params, source_params):
            # print("target_params:", targ.device)
            # print("source_params:", src.device)
            targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, 'module') else model


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)
