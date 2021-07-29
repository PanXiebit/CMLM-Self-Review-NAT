# CMLM-Self-Review NAT

Code for our COLING 2020 paper, "[Infusing Sequential Information into Conditional Masked Translation Model with Self-Review Mechanism](https://aclanthology.org/2020.coling-main.2.pdf)". Please cite our paper if you find this repository helpful in your research:

```
@inproceedings{Xie2020InfusingSI,
  title={Infusing Sequential Information into Conditional Masked Translation Model with Self-Review Mechanism},
  author={Pan Xie and Zhi Cui and Xiuying Chen and Xiaohui Hu and Jianwei Cui and Bin Wang},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics, {COLING} 2020, Barcelona, Spain (Online), December 8-13, 2020},
  publisher = {International Committee on Computational Linguistics},
  year={2020},
}
```


### Data Preprocess and Train

```
# process data

mkdir data

# WMT14 EN-DE
cd data
mkdir wmt14.en-de
cd wmt14.en-de
mkdir wmt16_en_de
cd wmt16_en_de
gdown https://drive.google.com/uc?id=0B_bZck-ksdkpM25jRUN2X2UxMm8
tar -zxvf wmt16_en_de.tar.gz
cd ..
cp wmt16_en_de/train.tok.clean.bpe.32000.en train.en
cp wmt16_en_de/train.tok.clean.bpe.32000.de train.de
cp wmt16_en_de/newstest2013.tok.bpe.32000.en valid.en
cp wmt16_en_de/newstest2013.tok.bpe.32000.de valid.de
cp wmt16_en_de/newstest2014.tok.bpe.32000.en test.en
cp wmt16_en_de/newstest2014.tok.bpe.32000.de test.de
rm -r wmt16_en_de
cd ../..
python preprocess.py --source-lang en --target-lang de --trainpref data/wmt14.en-de/train --validpref data/wmt14.en-de/valid --testpref data/wmt14.en-de/test --destdir output/data-bin/wmt14.en-de --joined-dictionary --workers 8 --nwordssrc 32768 --nwordstgt 32768
python preprocess.py --source-lang de --target-lang en --trainpref data/wmt14.en-de/train --validpref data/wmt14.en-de/valid --testpref data/wmt14.en-de/test --destdir output/data-bin/wmt14.de-en --joined-dictionary --workers 8 --nwordssrc 32768 --nwordstgt 32768
    
# train model

src=en
tgt=de
model_dir=output/my_maskPredict_${src}_${tgt}

python train.py output/data-bin/wmt14.${src}-${tgt} \
   --arch bert_transformer_seq2seq_gan \
   --share-all-embeddings \
   --criterion label_smoothed_length_gan_cross_entropy \
   --label-smoothing 0.1 \
   --lr 5e-4 \
   --warmup-init-lr 1e-7 \
   --min-lr 1e-9 \
   --lr-scheduler inverse_sqrt \
   --warmup-updates 10000 \
   --optimizer adam \
   --adam-betas '(0.9, 0.999)' \
   --adam-eps 1e-6 \
   --task translation_self \
   --max-tokens 1000 \
   --weight-decay 0.01 \
   --dropout 0.3 \
   --encoder-layers 6 \
   --encoder-embed-dim 512 \
   --decoder-layers 6 \
   --decoder-embed-dim 512 \
   --max-source-positions 10000 \
   --max-target-positions 10000 \
   --max-update 300000 \
   --seed 0 \
   --save-dir ${model_dir} \
   --dis_weights 5.0\
   --restore-file checkpoint_last.pt #> train-log 2>&1 &

```

### Evaluation

```
src=en
tgt=de
model_path=output
model_dir=${model_path}/my_maskPredict_${src}_${tgt}


python generate_cmlm.py \
    --data output/data-bin \
    --path ${model_dir}/checkpoint_average.pt \
    --task translation_self \
    --remove-bpe True \
    --max-sentences 20 \
    --decoding-iterations 1 \
    --decoding-strategy mask_predict
```

