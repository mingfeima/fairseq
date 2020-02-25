###
### script for inference iwslt14 german to english dataset
### reference
###   https://github.com/pytorch/fairseq/tree/master/examples/translation#iwslt14-german-to-english-transformer
###
### 1. prepare pre-trained model
###   the script will download the pre-trained model from my google drive
###   in case you have network connection issue, manually download it and place to right directory.
###
### 2. install:
###   pip install --editable .
###
### 3. cpu runs:
###   ./run_inference_cpu.sh (throughput)
###   ./run_inference_cpu.sh --single realtime)
###
### TODO:
###   a. low perf due to model modification
###   b. add mkldnn support (use MkldnnLinear)

### Please config jemalloc before running this script, this is crucial for CPU performance
###   1. jemalloc: https://github.com/jemalloc/jemalloc/wiki/Getting-Started
###      a) download from release: https://github.com/jemalloc/jemalloc/releases
###      b) tar -jxvf jemalloc-5.2.0.tar.bz2
###      c) ./configure
###         make
###      d) cd /home/mingfeim/packages/jemalloc-5.2.0/bin
###         chmod 777 jemalloc-config
###


export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";
export LD_PRELOAD=/home/mingfeim/packages/jemalloc-5.2.0/lib/libjemalloc.so


CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"


PREFIX=""
BATCH_SIZE=256
if [[ "$1" == "--single" ]]; then
  echo "### using single batch size"
  BATCH_SIZE=1
  TOTAL_CORES=4
  LAST_CORE=`expr $TOTAL_CORES - 1`
  PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"
  shift
fi

export OMP_NUM_THREADS=$CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$CORES"
echo -e "### using $KMP_SETTING"
sleep 3

MODEL=./data-bin/checkpoint_best.pt
if [ -e $MODEL ]; then
  echo "### $MODEL found..."
else
  echo "### $MODEL doesn't exist, prepare to download..."
  echo "### install gdown..."
  pip install gdown
  echo "### download model..."
  gdown  https://drive.google.com/uc?id=13-bTrj9nD5mhSuP_EpftZcJM3jShXU1C -O $MODEL
fi

LOG0=inference_cpu_bs${BATCH_SIZE}_NODE0.txt
numactl --cpunodebind=0 --membind=0 fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path $MODEL \
    --batch-size $BATCH_SIZE --beam 5 --remove-bpe \
    --cpu \
    2>&1 | tee $LOG0 &

LOG1=inference_cpu_bs${BATCH_SIZE}_NODE1.txt
numactl --cpunodebind=1 --membind=1 fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path $MODEL \
    --batch-size $BATCH_SIZE --beam 5 --remove-bpe \
    --cpu \
    2>&1 | tee $LOG1

echo -e "\n\n Sum sentences/s together:"
tail -n 2 $LOG0
tail -n 2 $LOG1
