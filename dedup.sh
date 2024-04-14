json_path="/home/ubuntu/Workspace/DB/korean_db/data/KtelSpeech/timestamped_train.json"
output_dir="Ktel_train"

mkdir -p $output_dir
python -m text_dedup.minhash_from_json \
  --path $json_path \
  --cache_dir "./cache" \
  --output ${output_dir} \
  --column "text" \
  --batch_size 1

cat ${output_dir}/final_data.json | sed '/^$/d' | sed 's/}/},/g' | sed 's#\\/#/#g' | sed '1i\[' | sed '$d' > ${output_dir}/temp.json
mv ${output_dir}/temp.json ${output_dir}/final_data.json
echo "    }" >> ${output_dir}/final_data.json
echo "]" >> ${output_dir}/final_data.json
