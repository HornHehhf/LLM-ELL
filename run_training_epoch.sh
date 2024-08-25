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

declare -A batch_sizes
batch_sizes[bookcorpus]=10
batch_sizes[c4]=1
batch_sizes[openwebtext]=1
batch_sizes[wiki]=1
batch_sizes[pes2o]=1
batch_sizes[pile]=1
batch_sizes[redpajama]=1
batch_sizes[oscar]=1

declare -A step_nums
step_nums[4b]=3814
step_nums[8b]=7629
step_nums[12b]=11444
step_nums[16b]=15258
step_nums[24b]=22888
step_nums[32b]=30517
step_nums[40b]=38146

for total_num in 4b 8b 12b 16b 24b 32b 40b
do
    for data in openwebtext
    do
        sent_num=${sent_nums[${data}]}
        batch_size=${batch_sizes[${data}]}
        step_num=${step_nums[${total_num}]}
        model=gpt2-2b8${total_num}4b_global_step${step_num}
        echo "save ${model} ${data}"
        CUDA_VISIBLE_DEVICES=0 python analyzing_law.py model=${model} phase=pretrained data=${data} sent_num=${sent_num} batch_size=${batch_size} feature=save regression=prediction-residual normalize=initialized-LN > logs/${model}_${data}_pretrained_features_size=${sent_num}.log 2>&1
        echo "load ${model} ${data}"
        CUDA_VISIBLE_DEVICES=0 python analyzing_law.py model=${model} phase=pretrained data=${data} sent_num=${sent_num} batch_size=${batch_size} feature=load regression=prediction-residual normalize=initialized-LN > logs/${model}_${data}_pretrained_features_size=${sent_num}.log 2>&1
    done
done

end=$(date +%s)
runtime=$((end - start))
echo "Running time: $runtime seconds"
