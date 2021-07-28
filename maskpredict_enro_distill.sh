# process data

mkdir data

# WMT16 EN-RO
cd data
gdown https://drive.google.com/uc?id=1YrAwCEuktG-iDVxtEW-FE72uFTLc5QMl
tar -zxvf wmt16.tar.gz

mkdir wmt16.distill.en-ro
mkdir wmt16.distill.ro-en

cp wmt16/en-ro/distill/enro/corpus.bpe.en  wmt16.distill.en-ro/train.en
cp wmt16/en-ro/distill/enro/corpus.bpe.ro  wmt16.distill.en-ro/train.ro
cp wmt16/en-ro/dev/dev.bpe.en  wmt16.distill.en-ro/valid.en
cp wmt16/en-ro/dev/dev.bpe.ro  wmt16.distill.en-ro/valid.ro
cp wmt16/en-ro/test/test.bpe.en  wmt16.distill.en-ro/test.en
cp wmt16/en-ro/test/test.bpe.ro  wmt16.distill.en-ro/test.ro

cp wmt16/en-ro/distill/roen/corpus.bpe.en  wmt16.distill.ro-en/train.en
cp wmt16/en-ro/distill/roen/corpus.bpe.ro  wmt16.distill.ro-en/train.ro
cp wmt16/en-ro/dev/dev.bpe.en  wmt16.distill.ro-en/valid.en
cp wmt16/en-ro/dev/dev.bpe.ro  wmt16.distill.ro-en/valid.ro
cp wmt16/en-ro/test/test.bpe.en  wmt16.distill.ro-en/test.en
cp wmt16/en-ro/test/test.bpe.ro  wmt16.distill.ro-en/test.ro

rm wmt16.tar.gz
cd ..

text=data/wmt16.distill.en-ro
output_dir=output
src=en
tgt=ro
model_path=output
python preprocess.py \
   --source-lang ${src} \
   --target-lang ${tgt} \
   --trainpref $text/train \
   --validpref $text/valid \
   --testpref $text/test \
   --destdir ${output_dir}/data-distill-${src}${tgt} \
   --workers 60 \
   --srcdict ${model_path}/my_maskPredict_${src}_${tgt}/dict.${src}.txt \
   --tgtdict ${model_path}/my_maskPredict_${src}_${tgt}/dict.${tgt}.txt
    
# # train model
model_dir=${model_path}/my_maskPredict_${src}_${tgt}

python train.py \
   --data "output/data-bin" \
   --arch bert_transformer_seq2seq_gan \
   --share-all-embeddings \
   --sharing_gen_dis \
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
   --max-tokens 6144 \
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
   --dis_weights 1.0 \
   --restore-file checkpoint_last.pt #> train-log 2>&1 &
