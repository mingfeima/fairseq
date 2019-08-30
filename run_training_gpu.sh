###
### script for training iwslt14 german to english dataset
### https://github.com/pytorch/fairseq/tree/master/examples/translation#iwslt14-german-to-english-transformer
###


LOG='train_gpu.txt'
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    | tee $LOG
