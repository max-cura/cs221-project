#!/bin/bash

PYTHON=python3
T5_LIAR_TRAIN=t5_liar_train.py
ROBERTA_LIAR_TRAIN=roberta_liar_train.py

LOG=train-log.txt

train_proxy() {
	FILE=$(mktemp)
	COMMAND="${PYTHON} ${@} > $FILE"
	echo $COMMAND
	TOKENIZERS_PARALLELISM=true $PYTHON $@ > $FILE

	echo "" >> $LOG
	echo "$PYTHON $@" >> $LOG
	cat $FILE | grep "BEST MODEL" >> $LOG
	tail -n1 $FILE >> $LOG

	tail -n4 $LOG
}

train_proxy $T5_LIAR_TRAIN \
	    --use-tokenizer t5-small --use-model t5-small --dataset liar \
	    --max-length 256 --train-batch-size 128 --interval 4 \
	    --examples 1000 --model-dir t5-small-liar-1k
train_proxy $T5_LIAR_TRAIN \
	    --use-tokenizer t5-small --use-model t5-small --dataset liar \
	    --max-length 256 --train-batch-size 128 --interval 6 \
	    --examples 5000 --model-dir t5-small-liar-5k
train_proxy $T5_LIAR_TRAIN \
	    --use-tokenizer t5-small --use-model t5-small --dataset liar \
	    --max-length 256 --train-batch-size 128 --interval 8 \
	    --examples 0 --model-dir t5-small-liar-full

train_proxy $T5_LIAR_TRAIN \
	    --use-tokenizer t5-base --use-model t5-base --dataset liar \
	    --max-length 256 --train-batch-size 64 --interval 4 \
	    --examples 1000 --model-dir t5-base-liar-1k
train_proxy $T5_LIAR_TRAIN \
	    --use-tokenizer t5-base --use-model t5-base --dataset liar \
	    --max-length 256 --train-batch-size 64 --interval 12 \
	    --examples 5000 --model-dir t5-base-liar-5k
train_proxy $T5_LIAR_TRAIN \
	    --use-tokenizer t5-base --use-model t5-base --dataset liar \
	    --max-length 256 --train-batch-size 64 --interval 32 \
	    --examples 0 --model-dir t5-base-liar-full
rain_proxy $T5_LIAR_TRAIN \
	    --use-tokenizer t5-large --use-model t5-large --dataset liar \
	    --max-length 256 --train-batch-size 16 --interval 64 \
	    --examples 1000 --model-dir t5-large-liar-1k
#train_proxy $T5_LIAR_TRAIN \
#	    --use-tokenizer t5-large --use-model t5-large --dataset liar \
#	    --max-length 256 --train-batch-size 16 --interval 64 \
#	    --examples 5000 --model-dir t5-large-liar-5k
#train_proxy $T5_LIAR_TRAIN \
#	    --use-tokenizer t5-large --use-model t5-large --dataset liar \
#	    --max-length 256 --train-batch-size 16 --interval 64 \
#	    --examples 0 --model-dir t5-large-liar-full

train_proxy $ROBERTA_LIAR_TRAIN \
       	    --use-tokenizer roberta-base --use-model roberta-base --dataset liar \
	    --max-length 256 --train-batch-size 64 --interval 4 \
	    --examples 1000 --model-dir roberta-base-liar-1k
train_proxy $ROBERTA_LIAR_TRAIN \
       	    --use-tokenizer roberta-base --use-model roberta-base --dataset liar \
	    --max-length 256 --train-batch-size 64 --interval 12 \
	    --examples 5000 --model-dir roberta-base-liar-5k
train_proxy $ROBERTA_LIAR_TRAIN \
       	    --use-tokenizer roberta-base --use-model roberta-base --dataset liar \
	    --max-length 256 --train-batch-size 64 --interval 32 \
	    --examples 0 --model-dir roberta-base-liar-full

#train_proxy $ROBERTA_LIAR_TRAIN \
#       	    --use-tokenizer roberta-large --use-model roberta-large --dataset liar \
#	    --max-length 256 --train-batch-size 32 --interval 4 \
#	    --examples 1000 --model-dir roberta-large-liar-1k
#train_proxy $ROBERTA_LIAR_TRAIN \
#       	    --use-tokenizer roberta-large --use-model roberta-large --dataset liar \
#	    --max-length 256 --train-batch-size 32 --interval 16 \
#	    --examples 5000 --model-dir roberta-large-liar-5k
#train_proxy $ROBERTA_LIAR_TRAIN \
#       	    --use-tokenizer roberta-large --use-model roberta-large --dataset liar \
#	    --max-length 256 --train-batch-size 32 --interval 32 \
#	    --examples 0 --model-dir roberta-large-liar-full
