import argparse
import os
import pickle
from contextlib import nullcontext

import opacus
import numpy as np
import torch
from datasets import load_dataset
from opacus.utils import batch_memory_manager
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BartForConditionalGeneration


def main():
    def tokenize_data(example):
        return tokenizer(example['src'] + example['trg'], padding='max_length', truncation=True, max_length=256,
                         add_special_tokens=True)

    def tokenize_data_cond(example):
        return tokenizer(example['src'], text_target=example['trg'], padding='max_length', truncation=True, max_length=256,
                         add_special_tokens=True)

    # flags for training run
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Name of the model')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--epsilon', type=int, default=1, help='Epsilon, accepts integers')
    parser.add_argument('--private', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)

    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset('json', data_files=os.path.join('..', 'data', args.dataset, 'train.jsonl'))['train']

    # todo debug mode
    # dataset = dataset['train'].select(range(100))

    # Select model
    if args.model == 'bloom':
        model_name = 'bigscience/bloom-560m'
        max_phys = 8
    elif args.model == 'phi':
        max_phys = 4
        model_name = 'microsoft/phi-1_5'
    elif args.model == 'bart':
        max_phys = 8
        model_name = 'facebook/bart-large'
    else:
        raise NotImplementedError('Experiments only implemented for the models selected for the paper.')

    # Load model, opt and tokenizer
    if args.model != 'bart':
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # preprocess
    print("Started preprocessing ...")
    if args.model != 'bart':
        data_loader = torch.utils.data.DataLoader(
            dataset.map(tokenize_data, remove_columns=dataset.column_names).with_format('torch'),
            batch_size=args.batch_size,
            num_workers=0
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset.map(tokenize_data_cond, remove_columns=dataset.column_names).with_format('torch'),
            batch_size=args.batch_size,
            num_workers=0
        )

    print("Started training ...")
    # determine if DP run
    if args.private:
        privacy_engine = opacus.PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            target_delta=(len(dataset) * 10) ** -1,
            target_epsilon=args.epsilon,
            epochs=args.epochs,
            max_grad_norm=1.0
        )
        # freeze opacus-incompatible layers
        if args.model == 'bart':
            model._module.base_model.decoder.embed_positions.weight.requires_grad = False
        # create memory safe context for dp-sgd training

        context_manager = batch_memory_manager.BatchMemoryManager(
            data_loader=data_loader,
            max_physical_batch_size=max_phys,
            optimizer=optimizer
        )
    else:
        context_manager = nullcontext()

    with context_manager as train_dataloader:
        if not args.private:
            train_dataloader = data_loader

        pbar = tqdm(total=args.epochs * len(train_dataloader), position=0)
        step = 0

        # Train loop
        for epoch in range(0, args.epochs):
            losses = []

            for batch in train_dataloader:
                optimizer.zero_grad()
                inputs = {k: v.to('cuda') for k, v in batch.items()}

                if args.model != 'bart':
                    outputs = model(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"],
                                    labels=inputs["input_ids"])
                else:
                    outputs = model(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"],
                                    labels=inputs["labels"])
                loss = outputs.loss
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                step += 1
                pbar.update(1)

                if step % 100 == 0:
                    pbar.set_description("Avg. loss: " + str(np.average(losses[-100:])))

                if step % 5000 == 0:
                    file_path = os.path.join('..', 'models', args.dataset,
                                             "eps{}".format(args.epsilon) if args.private else "np", args.model,
                                             f"{step:06d}")
                    if args.private:
                        model._module.save_pretrained(file_path)
                    else:
                        model.save_pretrained(file_path)





        # save again after training finished
        file_path = os.path.join('..', 'models', args.dataset,
                                 "eps{}".format(args.epsilon) if args.private else "np", args.model,
                                 f"{step:06d}")

        if args.private:
            model._module.save_pretrained(file_path)
        else:
            model.save_pretrained(file_path)

        with open(os.path.join('..', 'models', args.dataset,
                                 "eps{}".format(args.epsilon) if args.private else "np", args.model,
                                 "losses.pkl"), 'wb') as f:
            pickle.dump(losses, f)




if __name__ == '__main__':
    main()
