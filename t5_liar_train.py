import os

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, DataCollatorForSeq2Seq, \
    AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from datasets import load_dataset
from argparse import ArgumentParser
#import evaluate

torch.backends.cuda.matmul.allow_tf32 = True

argp = ArgumentParser()
argp.add_argument("--use-tokenizer", required=True)
argp.add_argument("--dataset", required=True)
#argp.add_argument("--use-train-dataset", required=True)
#argp.add_argument("--use-test-dataset", required=True)
argp.add_argument("--use-model", required=True)
argp.add_argument("--model-dir", required=True)
#argp.add_argument("--metric", default="")
argp.add_argument("--train-batch-size", type=int, default=8)
#argp.add_argument("--test-batch-size", type=int, default=8)
argp.add_argument("--max-length", type=int, default=1024)
argp.add_argument("--interval", type=int, required=True)
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
#dataset = load_dataset(
#    'csv',
#    data_files={
#        'train': args.use_train_dataset,
#        'test': args.use_test_dataset,
#    },
#    sep="\t",
#    on_bad_lines='skip',
#)
dataset = load_dataset(args.dataset)
print(dataset)
#if dataset["train"].num_columns != 2:
#    raise RuntimeError(f"Failed to load USE_TRAIN_DATASET <{args.use_train_dataset}>: wrong number of columns: expected 2 got {dataset['train'].num_columns}")
#if dataset["test"].num_columns != 2:
#    raise RuntimeError(f"Failed to load USE_TEST_DATASET <{args.use_test_dataset}>: wrong number of columns: expected 2 got {dataset['test'].num_columns}")
#if 'input' not in dataset["train"].column_names or 'output' not in dataset["train"].column_names:
#    raise RuntimeError(f"Failed to load USE_TRAIN_DATASET <{args.use_train_dataset}>: wrong column names: expected 'input' and 'output', got {dataset.column_names.keys()}")
#if 'input' not in dataset["test"].column_names or 'output' not in dataset["test"].column_names:
#    raise RuntimeError(f"Failed to load USE_TEST_DATASET <{args.use_test_datset}>: wrong column names: expected 'input' and 'output', got {dataset.column_names.keys()}")

#dataset = dataset.filter(lambda x: x["input"] is not None and x["output"] is not None)

def ds_tokenize(tokenizer, max_length, text=None, text_target=None):
    return tokenizer(
        text=text,
        text_target=text_target,
        # padding="max_length",
        truncation=True,
        # return_tensors="pt"
        max_length=max_length,
    )
def ds_local_tokenize_for_training(examples):
    # statement, subject, speaker, job_title, state_info, party_affiliation, context
    #text_input = f"liar:stmt:{example['statement']}|subj:{example['subject']}|spkr:{example['speaker']}|job:{example['job_title']}|stat:{example['state_info']}|part:{example['party_affiliation']}|ctx:{example['context']}"
    text_input = [
        f"liar:stmt:{statement}|subj:{subject}|spkr:{speaker}|job:{job_title}|stat:{state_info}|part:{party_affiliation}|ctx:{context}"
        for statement,subject,speaker,job_title,state_info,party_affiliation,context
        in zip(examples['statement'],examples['subject'],examples['speaker'],
            examples['job_title'],examples['state_info'],examples['party_affiliation'],
            examples['context'])
    ]
    #print(text_input[0])
    #iids = ds_tokenize(tokenizer, args.max_length, text=text_input[0]).input_ids
    #print(tokenizer.decode(iids))
    #quit()
    #text_input = examples['statement']
    #print(examples['label'])
    label = ["true" if label in [0, 5, 4] else "false" for label in examples["label"]]

    model_inputs = ds_tokenize(tokenizer, text=text_input, max_length=args.max_length)
    labels = ds_tokenize(tokenizer, text_target=label, max_length=20)
    model_inputs["labels"] = labels['input_ids']
    #model_inputs["prompt"] = text_input
    #print(model_inputs)
    #print(model_inputs.keys())
    return model_inputs

print("="*60)
print("TOKENIZING")

dataset = dataset.map(ds_local_tokenize_for_training, batched=True, remove_columns=['label'])

#longest_item = max(dataset['train']['prompt'], key=lambda x: len(x))
#iids = ds_tokenize(tokenizer, args.max_length, text=longest_item).input_ids
#print(longest_item)
#print(len(iids))

trainer_args = Seq2SeqTrainingArguments(
    output_dir=args.model_dir,
    evaluation_strategy="steps",
    eval_steps=args.interval,
    logging_strategy="steps",
    logging_steps=args.interval,
    save_strategy="steps",
    save_steps=args.interval,
    # For AdamW on T5, use 3e-4 instead of e.g. 4e-5
    learning_rate=3e-4,
    #learning_rate=4e-5,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.train_batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    report_to=["tensorboard"],
    #gradient_accumulation_steps=4,
    #gradient_checkpointing=True,
    #use_cache=False,
    #optim="adafactor",
    dataloader_num_workers=8,
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

trainer = Seq2SeqTrainer(
    model_init=lambda: T5ForConditionalGeneration.from_pretrained(args.use_model),
    args=trainer_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("")
print("=" * 60)
print("TRAINING")

trainer.train()
