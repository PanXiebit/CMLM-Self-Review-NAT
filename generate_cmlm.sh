output_dir=output
src=en
tgt=de
model_path=output
model_dir=${model_path}/my_maskPredict_${src}_${tgt}


# python generate_cmlm.py output/data-bin/wmt14.${src}-${tgt} \
#     --path ${model_dir}/checkpoint5.pt \
#     --task translation_self \
#     --remove-bpe \
#     --max-sentences 20 \
#     --decoding-iterations 15 \
#     --decoding-strategy mask_predict

# mkdir generate_fuse_log
for i in {15..19}
do
  python generate_cmlm.py output/data-bin/wmt14.${src}-${tgt}  --path output/my_maskPredict_en_ro/checkpoint${i}.pt --task translation_self --remove-bpe --max-sentences 20 --decoding-iterations 15 --decoding-strategy mask_predict > generate_fuse_log/epoch_${i}_iter_15.log
  echo "Finished Epoch ${i}."
done
