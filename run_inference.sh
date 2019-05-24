#####################################################
## script to evaluate fairseq-transformer peformance
##   using multi instance
##   mingfei.ma@intel.com
#####################################################

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`


### global settings
# comment this line if you intend to use all cores
TOTAL_CORES=4

NUM_PROCESSES=$TOTAL_CORES
BATCH_SIZE=1
SENTENCES=100


# uncomment this to enable profiling
ARGS=""
#ARGS="$ARGS --profiling"


OMP_NUM_THREADS=$TOTAL_CORES fairseq-generate \
    data-bin/wmt14.en-fr.joined-dict.newstest2014 \
    --path data-bin/wmt14.en-fr.joined-dict.transformer/model.pt \
    --beam 5 --batch-size $BATCH_SIZE --remove-bpe --quiet \
    --num-processes=$NUM_PROCESSES --max-updates=$SENTENCES \
    $ARGS
