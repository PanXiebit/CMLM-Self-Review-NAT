# process data

# mkdir data

# # WMT16 EN-RO
# cd data
# mkdir wmt16.en-ro
# cd wmt16.en-ro
# gdown https://drive.google.com/uc?id=1YrAwCEuktG-iDVxtEW-FE72uFTLc5QMl
# tar -zxvf wmt16.tar.gz
# mv wmt16/en-ro/train/corpus.bpe.en train.en
# mv wmt16/en-ro/train/corpus.bpe.ro train.ro
# mv wmt16/en-ro/dev/dev.bpe.en valid.en
# mv wmt16/en-ro/dev/dev.bpe.ro valid.ro
# mv wmt16/en-ro/test/test.bpe.en test.en
# mv wmt16/en-ro/test/test.bpe.ro test.ro
# rm wmt16.tar.gz
# rm -r wmt16
# cd ../..

text=data/wmt16.en-ro
output_dir=output
src=en
tgt=ro
model_path=output
# python preprocess.py \
#     --source-lang ${src} \
#     --target-lang ${tgt} \
#     --trainpref $text/train \
#     --validpref $text/valid \
#     --testpref $text/test \
#     --destdir ${output_dir}/data-bin \
#     --workers 60 \
#     --srcdict ${model_path}/my_maskPredict_${src}_${tgt}/dict.${src}.txt \
#     --tgtdict ${model_path}/my_maskPredict_${src}_${tgt}/dict.${tgt}.txt
    
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
