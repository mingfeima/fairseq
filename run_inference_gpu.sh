###
### script for inference iwslt14 german to english dataset
### https://github.com/pytorch/fairseq/tree/master/examples/translation#iwslt14-german-to-english-transformer
###

BATCH_SIZE=128
if [[ "$1" == "--single" ]]; then
  echo "### using single batch size"
  BATCH_SIZE=1
  shift
fi

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

LOG=inference_gpu_bs${BATCH_SIZE}.txt
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path $MODEL \
    --batch-size $BATCH_SIZE --beam 5 --remove-bpe \
    2>&1 | tee $LOG
