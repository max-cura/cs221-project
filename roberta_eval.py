import argparse
import torch
from tqdm import tqdm
import transformers
from transformers import RobertaForSequenceClassification, GenerationConfig, RobertaTokenizerFast
#import torch.utils.data.DataLoader
from torch.utils.data import DataLoader
import evaluate
from tabulate import tabulate
import datasets

import os

# ARGUMENTS

argp = argparse.ArgumentParser()
argp.add_argument("--use-tokenizer", required=True)
argp.add_argument("--model", required=True)
argp.add_argument("--max-length", type=int, default=512)
argp.add_argument("--batch-size", type=int, default=64)
argp.add_argument("--dataset", required=True)
args = argp.parse_args()

# DEVICE

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# DATASETS

def tokenize_generic(tokenizer, max_length, text=None, text_target=None):
    return tokenizer(
        text=text,
        text_target=text_target,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )
def LIAR_prepare_dataset():
    def LIAR_batch_tokenize_test(examples):
        text_input = [
            f"liar:stmt:{statement}|subj:{subject}|spkr:{speaker}|job:{job_title}|stat:{state_info}|part:{party_affiliation}|ctx:{context}"
            for statement,subject,speaker,job_title,state_info,party_affiliation,context
            in zip(examples['statement'],examples['subject'],examples['speaker'],
                examples['job_title'],examples['state_info'],examples['party_affiliation'],
                examples['context'])
        ]
        label = [0 if label in [0, 5, 4] else 1 for label in examples["gold"]]

        model_inputs = tokenize_generic(tokenizer, text=text_input, max_length=args.max_length)
        #labels = tokenize_generic(tokenizer, text_target=label, max_length=20)
        #model_inputs["labels"] = labels['input_ids']
        model_inputs["gold_tf"] = label
        return model_inputs
    dataset = datasets.load_dataset('liar')

    dataset = dataset.rename_column('label', 'gold')
    dataset = dataset.map(LIAR_batch_tokenize_test, batched=True)
    return dataset

DATASETS = {
    'liar': LIAR_prepare_dataset,
}

# TOKENIZER & MODEL CREATION

print("=" * 60)
print("Initializing")

tokenizer = RobertaTokenizerFast.from_pretrained(
    args.use_tokenizer,
    model_max_length=args.max_length,
    padding=True,
    truncation=True,
)

if not os.path.isdir(args.model):
    raise ValueError(f"bad path for model")
model = RobertaForSequenceClassification.from_pretrained(args.model).to(device)

if args.dataset not in DATASETS:
    raise ValueError(f"bad dataset")
dataset = DATASETS[args.dataset]()

print(dataset)

device_dataset = dataset.with_format('torch')
predictions = torch.zeros(
        (device_dataset['test']['input_ids'].shape[0]),
        dtype=torch.int64
)
def batched_range_iter(start, end, batch_size):
    batch_start = start
    while batch_start < end:
        yield batch_start, min(batch_start+batch_size, end)
        batch_start += batch_size
print("=" * 60)
print("Generating predictions")
with torch.no_grad():
    batch_count = (len(device_dataset['test']) + args.batch_size - 1) // args.batch_size
    for batch_start, batch_end in tqdm(
            batched_range_iter(0, len(device_dataset['test']), batch_size=args.batch_size),
            total=batch_count):
        batch_inputs = device_dataset['test']['input_ids'][batch_start:batch_end].to(device)
        batch_attnmsk = device_dataset['test']['attention_mask'][batch_start:batch_end].to(device)
        batch_logits = model(
            input_ids=batch_inputs,
            attention_mask=batch_attnmsk
        ).logits
        predicted_classes = batch_logits.argmax(dim=1)
        #print(predicted_classes)
        predictions[batch_start:batch_end] \
            = predicted_classes.type(dtype=torch.int)

print("")
print("="*60)
print("Calculating confusion matrix")

dataset['test'] = dataset['test'].add_column('predicted', predictions.tolist())

def evaluate_on_ds(dataset):
    TP=0
    FP=0
    TN=0
    FN=0
    #print(dataset)
    true=1
    false=0
    for (pred,gold) in zip(dataset['test']['predicted'], dataset['test']['gold_tf']):
        #print(f"P:{pred},G:{gold}")
        if pred==true and gold==true:
            TP += 1
        elif pred==true and gold==false:
            FP += 1
        elif pred==false and gold==true:
            FN += 1
        elif pred==false and gold==false:
            TN += 1

    print( "RESULT Confusion matrix")
    print( "RESULT " + "="*30)
    S = TP+FP+TN+FN
    def confusion_matrix(tp,fn,fp,tn):
        s = tabulate(
                [['AP',tp,fn,tp+fn], ['AN', fp, tn,fp+tn], ['',tp+fp,fn+tn,'']],
                headers = ['', 'PP','PN',''],
                tablefmt='orgtbl'
        )
        s = "RESULT " + s.replace("\n", "\nRESULT ")
        print(s)
    print("\nRESULT ---> RAW")
    confusion_matrix(TP,FN,FP,TN)
    print("\nRESULT ---> PROP")
    confusion_matrix(TP/S,FN/S,FP/S,TN/S)

all_datasets = [
('all', dataset),
('true', dataset.filter(lambda x: x["gold"]==3)),
('mostly_true', dataset.filter(lambda x: x['gold']==2)),
('half_true', dataset.filter(lambda x: x["gold"]==1)),
('barely_true', dataset.filter(lambda x: x["gold"]==4)),
('false', dataset.filter(lambda x: x["gold"]==0)),
('pants_fire', dataset.filter(lambda x: x["gold"]==5))
]
print(f"RESULT FOR {args.model}")
for name, ds in all_datasets:
    print(f"RESULT ON {name} DATA:")
    save_path = f"{args.model[:args.model.find('/')]}-test-predict.csv"
    print(f"RESULT RAW DATA TO: {save_path}")
    ds['test'].to_csv(save_path,index=None)
    evaluate_on_ds(ds)

#s="Says the Annies List political group supports third-trimester abortions on demand."
#s="Building a wall on the U.S.-Mexico border will take literally years."
#s="Wisconsin is on pace to double the number of layoffs this year."
#iids = tokenizer(s, return_tensors="pt",padding="max_length",max_length=args.max_length, truncation=True).input_ids.to(device)
#outs = model.generate(input_ids = iids)
#print(f"prompt: {s}")
#print(f"result: {tokenizer.decode(outs[0], skip_special_tokens=True)}")



