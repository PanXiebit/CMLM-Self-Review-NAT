# process data

src=$1
tgt=$2
data_path=data-distill/distill.wmt14.${src}-${tgt}

python preprocess.py \
  --source-lang ${src} \
  --target-lang ${tgt} \
  --trainpref ${data_path}/train \
  --validpref ${data_path}/valid \
  --testpref ${data_path}/test \
  --destdir output/data-bin/distill.wmt14.${src}-${tgt} \
  --joined-dictionary \
  --workers 8 \
  --nwordssrc 32768 
  --nwordstgt 32768

# train model

model_dir=output/my_distill_maskPredict_${src}_${tgt}

python train.py \
   --data output/data-bin/distill.wmt14.${src}-${tgt} \
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
   --max-tokens 6144 \
   --weight-decay 0.01 \
   --dropout 0.3 \
   --encoder-layers 6 \
   --encoder-embed-dim 512 \
   --decoder-layers 6 \
   --decoder-embed-dim 512 \
   --max-source-positions 10000 \
   --max-target-positions 10000 \
   --max-update 3000000 \
   --seed 0 \
   --save-dir ${model_dir} \
   --dis_weights 1.0 \
   --validate-interval 5 \
   --update-freq 2\
   --restore-file checkpoint_last.pt #> train-log 2>&1 &

