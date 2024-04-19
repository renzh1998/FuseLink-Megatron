#!/bin/bash
set -ex

BASE_PATH=dataset/Megatron-LM
DATA_PATH=${BASE_PATH}/arxiv_text_document/arxiv_text_document
DS_CONFIG=ds_config.json

TP=1
PP=4
NLAYERS=24
HIDDEN=1024

SEQLEN=256
GLOBAL_BATCH=64
MICRO_BATCH=16

ZERO_STAGE=0

current_time=$(date "+%m.%d-%H.%M.%S")
OUTPUT_PATH=gpt_output
OUTPUT_DIR=$OUTPUT_PATH/ds_${current_time}_z${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
#OUTPUT_DIR=baseline_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
mkdir -p $OUTPUT_DIR

GPUS_PER_NODE=4
MASTER_ADDR='192.168.1.149'
MASTER_PORT=6001
NNODES=2
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# export NCCL_DEBUG=INFO 
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
      pretrain_gpt.py \
    --num-workers 2 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --no-masked-softmax-fusion \
    --recompute-method uniform \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads 16 \
    --seq-length $SEQLEN \
    --loss-scale 12 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters 3 \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 2 \
    --eval-interval 1000 \
    --data-path $DATA_PATH \
    --vocab-file $BASE_PATH/gpt2-vocab.json \
    --merge-file $BASE_PATH/gpt2-merges.txt \
    --save-interval 1000 \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --fp16 \
    --checkpoint-activations \
    --tensorboard-dir $OUTPUT_DIR \
    --exit-interval 5000 | tee ${OUTPUT_DIR}/output.log

