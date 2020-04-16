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

# change this number to adjust number of instances
CORES_PER_INSTANCE=4


KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"


PREFIX=""
BATCH_SIZE=1

export OMP_NUM_THREADS=$CORES_PER_INSTANCE
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$CORES_PER_INSTANCE"
echo -e "### using $KMP_SETTING\n\n"
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


INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
LAST_INSTANCE=`expr $INSTANCES - 1`
INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`
for i in $(seq 1 $LAST_INSTANCE); do
    numa_node_i=`expr $i / $INSTANCES_PER_SOCKET`
    start_core_i=`expr $i \* $CORES_PER_INSTANCE`
    end_core_i=`expr $start_core_i + $CORES_PER_INSTANCE - 1`
    LOG_i=inference_cpu_bs${BATCH_SIZE}_ins${i}.txt

    echo "### running on instance $i, numa node $numa_node_i, core list {$start_core_i, $end_core_i}..."
    numactl --physcpubind=$start_core_i-$end_core_i --membind=$numa_node_i fairseq-generate data-bin/iwslt14.tokenized.de-en \
        --path $MODEL \
        --batch-size $BATCH_SIZE --beam 5 --remove-bpe \
        --cpu \
        2>&1 | tee $LOG_i &
done


numa_node_0=0
start_core_0=0
end_core_0=`expr $CORES_PER_INSTANCE - 1`
LOG_0=inference_cpu_bs${BATCH_SIZE}_ins0.txt

echo "### running on instance 0, numa node $numa_node_0, core list {$start_core_0, $end_core_0}...\n\n"
numactl --physcpubind=$start_core_0-$end_core_0 --membind=$numa_node_0 fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path $MODEL \
    --batch-size $BATCH_SIZE --beam 5 --remove-bpe \
    --cpu \
    2>&1 | tee $LOG_0

sleep 10
echo -e "\n\n Sum sentences/s together:"
for i in $(seq 0 $LAST_INSTANCE); do
    log=inference_cpu_bs${BATCH_SIZE}_ins${i}.txt
    tail -n 2 $log
done
