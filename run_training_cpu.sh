###
### CPU training script for TransformerLT
### 
### 0.
### 1. dataset: wmt17-eng2de:
###    https://github.com/pytorch/fairseq/tree/master/examples/translation#wmt14-english-to-german-convolutional
###

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export LD_PRELOAD=/home/mingfeim/packages/jemalloc-5.2.0/lib/libjemalloc.so

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
LAST_CORE=`expr $CORES - 1`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

SOCKET=`expr $SOCKETS - 1`
START_CORE=`expr $SOCKET \* $CORES`
END_CORE=`expr $START_CORE + $CORES - 1`
PREFIX="numactl --physcpubind=$START_CORE-$END_CORE --membind=$SOCKET"

export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo -e "\n### using $KMP_SETTING"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME\n"

### single socket
echo -e "\n### using OMP_NUM_THREADS=$CORES"
echo -e "### using $PREFIX\n"
OMP_NUM_THREADS=$CORES $PREFIX fairseq-train data-bin/wmt17_en_de/ \
    --arch transformer_wmt_en_de_big \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --num-workers 0 \
    --cpu

