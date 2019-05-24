```bash
# retrieve source code
git clone https://github.com/mingfeima/fairseq
cd fairseq
git checkout multi_instance_inference

# prepare dataset
mkdir -p data-bin
curl https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2 | tar xvjf - -C data-bin
curl https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2 | tar xvjf - -C data-bin

# run
./run_inference.sh
```
