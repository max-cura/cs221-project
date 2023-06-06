import os

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, \
        TrainingArguments
from datasets import load_dataset, Features, ClassLabel
from argparse import ArgumentParser

torch.backends.cuda.matmul.allow_tf32 = True

argp = ArgumentParser()
argp.add_argument("--use-tokenizer", required=True)
argp.add_argument("--dataset", required=True)
argp.add_argument("--use-model", required=True)
argp.add_argument("--model-dir", required=True)
argp.add_argument("--train-batch-size", type=int, default=8)
argp.add_argument("--max-length", type=int, default=1024)
argp.add_argument("--interval", type=int, required=True)
argp.add_argument("--examples", type=int, default=0)
args = argp.parse_args()

# DEVICE

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# TOKENIZER

tokenizer = AutoTokenizer.from_pretrained(
    args.use_tokenizer,
    model_max_length=args.max_length,
    use_fast=True,
)

# DATASET

print()
print('='*60)
print(f'LOADING AND FILTERING')
print(f'\tDATASET={args.dataset}')

dataset = load_dataset(args.dataset)
print(dataset)

def ds_tokenize(tokenizer, max_length, text=None, text_target=None):
    return tokenizer(
        text=text,
        text_target=text_target,
        # padding="max_length",
        truncation=True,
        #return_tensors="pt",
        max_length=max_length,
    )
def ds_local_tokenize_for_training(examples):
    text_input = [
        f"liar:stmt:{statement}|subj:{subject}|spkr:{speaker}|job:{job_title}|stat:{state_info}|part:{party_affiliation}|ctx:{context}"
        for statement,subject,speaker,job_title,state_info,party_affiliation,context
        in zip(examples['statement'],examples['subject'],examples['speaker'],
            examples['job_title'],examples['state_info'],examples['party_affiliation'],
            examples['context'])
    ]
    label = [0 if label in [0, 5, 4] else 1 for label in examples["gold_mc"]]

    model_inputs = ds_tokenize(tokenizer, text=text_input, max_length=args.max_length)
    model_inputs['labels'] = label
    return model_inputs

print("="*60)
print("TOKENIZING")

# ROBERTA
dataset = dataset.rename_column('label','gold_mc')
dataset = dataset.map(ds_local_tokenize_for_training, batched=True)
dataset = dataset.cast_column('labels', ClassLabel(num_classes=2,names=['false','true']))

#longest_item = max(dataset['train']['prompt'], key=lambda x: len(x))
#iids = ds_tokenize(tokenizer, args.max_length, text=longest_item).input_ids
#print(longest_item)
#print(len(iids))

training_data = dataset['train']
validation_data = dataset['validation']

if args.examples > 0:
    print("="*60)
    print(f"SELECTING {args.examples} ROWS FROM TRAINING DATASET")
    training_data = training_data.select(range(args.examples))

trainer_args = TrainingArguments(
    output_dir=args.model_dir,
    evaluation_strategy="steps",
    eval_steps=args.interval,
    logging_strategy="steps",
    logging_steps=args.interval,
    save_strategy="steps",
    save_steps=args.interval,
    # orig: 4e-5
    learning_rate=2e-5,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.train_batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    report_to=["tensorboard"],
    dataloader_num_workers=8,
)


trainer = Trainer(
    model_init=lambda: RobertaForSequenceClassification.from_pretrained(
        args.use_model,
        num_labels=2
    ),
    args=trainer_args,
    train_dataset=training_data,
    eval_dataset=validation_data,
    tokenizer=tokenizer,
)

print("")
print("=" * 60)
print("TRAINING")

trainer.train()

print()
print("="*60)
print(f"BEST MODEL: {trainer.state.best_model_checkpoint}")
print(trainer.evaluate(validation_data))
