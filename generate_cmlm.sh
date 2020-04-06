output_dir=output
src=en
tgt=ro
model_path=output

model_dir=${model_path}/my_maskPredict_${src}_${tgt}


python generate_cmlm.py ${output_dir}/data-bin \
    --path ${model_dir}/checkpoint22.pt \
    --task translation_self \
    --remove-bpe \
    --max-sentences 20 \
    --decoding-iterations 1 \
    --decoding-strategy mask_predict
