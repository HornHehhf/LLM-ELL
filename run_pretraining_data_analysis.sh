#!/bin/bash
start=$(date +%s)

declare -A sent_nums
sent_nums[bookcorpus]=3000
sent_nums[c4]=200
sent_nums[openwebtext]=100
sent_nums[wiki]=100
sent_nums[pes2o]=100
sent_nums[pile]=100
sent_nums[redpajama]=100
sent_nums[oscar]=100

for model in gpt2-2b855b55boscar_global_step52452 gpt2-4b284b84boscar_global_step80108 gpt2-8b7178b178boscar_global_step84877 gpt2-2b855b55bc4_global_step52452 gpt2-4b284b84bc4_global_step80108 gpt2-8b7178b178b_global_step84877
do
    for data in openwebtext
    do
        sent_num=${sent_nums[${data}]}
        echo "save ${model} ${data}"
        python analyzing_law.py model=${model} phase=pretrained data=${data} sent_num=${sent_num} batch_size=1 feature=save regression=prediction-residual normalize=initialized-LN > logs/${model}_${data}_pretrained_features_size=${sent_num}.log 2>&1
        echo "load ${model} ${data}"
        python analyzing_law.py model=${model} phase=pretrained data=${data} sent_num=${sent_num} batch_size=1 feature=load regression=prediction-residual normalize=initialized-LN > logs/${model}_${data}_pretrained_features_size=${sent_num}.log 2>&1
    done
done


end=$(date +%s)
runtime=$((end - start))
echo "Running time: $runtime seconds"
