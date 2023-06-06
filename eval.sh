#!/bin/bash

PYTHON=python3
T5_EVAL=t5_eval.py
ROBERTA_EVAL=roberta_eval.py

TRAIN_LOG=train-log.txt
EVAL_LOG=eval-log.txt

eval_model() {
	FILE=$(mktemp)
	echo "$PYTHON $@ > $FILE"
	TOKENIZERS_PARALLELISM=true $PYTHON $@ > $FILE 2>/dev/null
	cat $FILE | grep "RESULT " >> $EVAL_LOG
	cat $FILE | grep "RESULT "
}

lookup_model() {
	grep "BEST MODEL: $1" $TRAIN_LOG | sed 's/BEST MODEL: //'
}

#eval_model $T5_EVAL --model $(lookup_model t5-small-liar-1k) \
#		    --use-tokenizer t5-small --max-length 256 --batch-size 256 \
#		    --dataset liar
#eval_model $T5_EVAL --model $(lookup_model t5-small-liar-5k) \
#		    --use-tokenizer t5-small --max-length 256 --batch-size 256 \
#		    --dataset liar
eval_model $T5_EVAL --model $(lookup_model t5-small-liar-full) \
		    --use-tokenizer t5-small --max-length 256 --batch-size 256 \
		    --dataset liar

#eval_model $T5_EVAL --model $(lookup_model t5-base-liar-1k) \
#		    --use-tokenizer t5-base --max-length 256 --batch-size 256 \
#		    --dataset liar
#eval_model $T5_EVAL --model $(lookup_model t5-base-liar-5k) \
#		    --use-tokenizer t5-base --max-length 256 --batch-size 256 \
#		    --dataset liar
#eval_model $T5_EVAL --model $(lookup_model t5-base-liar-full) \
#		    --use-tokenizer t5-base --max-length 256 --batch-size 256 \
#		    --dataset liar
#
#eval_model $T5_EVAL --model $(lookup_model t5-large-liar-1k) \
#		    --use-tokenizer t5-large --max-length 256 --batch-size 256 \
#		    --dataset liar
#
#eval_model $ROBERTA_EVAL --model $(lookup_model roberta-base-liar-1k) \
#		    --use-tokenizer roberta-base --max-length 256 --batch-size 256 \
#		    --dataset liar
#eval_model $ROBERTA_EVAL --model $(lookup_model roberta-base-liar-5k) \
#		    --use-tokenizer roberta-base --max-length 256 --batch-size 256 \
#		    --dataset liar
#eval_model $ROBERTA_EVAL --model $(lookup_model roberta-base-liar-full) \
#		    --use-tokenizer roberta-base --max-length 256 --batch-size 256 \
#		    --dataset liar

#eval_model $ROBERTA_EVAL --model $(lookup_model roberta-large-liar-1k) \
#		    --use-tokenizer roberta-large --max-length 256 --batch-size 256 \
#		    --dataset liar
#eval_model $ROBERTA_EVAL --model $(lookup_model roberta-large-liar-5k) \
#		    --use-tokenizer roberta-large --max-length 256 --batch-size 256 \
#		    --dataset liar
#eval_model $ROBERTA_EVAL --model $(lookup_model roberta-large-liar-full) \
#		    --use-tokenizer roberta-large --max-length 256 --batch-size 256 \
#		    --dataset liar
#
