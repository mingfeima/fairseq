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


CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"


PREFIX=""
BATCH_SIZE=128
if [[ "$1" == "--single" ]]; then
  echo "### using single batch size"
  BATCH_SIZE=1
  TOTAL_CORES=4
  LAST_CORE=`expr $TOTAL_CORES - 1`
  PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"
  shift
fi

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
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

LOG=profile_cpu_bs${BATCH_SIZE}.txt
$PREFIX fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path $MODEL \
    --batch-size $BATCH_SIZE --beam 5 --remove-bpe \
    --cpu \
    --profile \
    2>&1 | tee $LOG
