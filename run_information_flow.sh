#!/bin/bash
start=$(date +%s)

declare -A probing_data
probing_data[gpt2-xl]=bookcorpus
probing_data[Meta-Llama-3-8B-Instruct]=bookcorpus
probing_data[rwkv-raven-14b]=c4
probing_data[mamba-2.8b-hf]=bookcorpus

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


for model in gpt2-xl Meta-Llama-3-8B-Instruct rwkv-raven-14b mamba-2.8b-hf
do
    data=${probing_data[${model}]}
    sent_num=${sent_nums[${data}]}
    batch_size=${batch_sizes[${data}]}
    for token in 'previous' 'current' 'next' 'next-next'
    do
        echo "save ${model} ${data}"
        python analyzing_law.py model=${model} phase=pretrained data=${data} sent_num=${sent_num} batch_size=${batch_size} feature=save regression=prediction-residual normalize=initialized-LN token=${token} > logs/${model}_${data}_pretrained_features_size=${sent_num}_${token}.log 2>&1
        echo "load ${model} ${data}"
        python analyzing_law.py model=${model} phase=pretrained data=${data} sent_num=${sent_num} batch_size=${batch_size} feature=load regression=prediction-residual normalize=initialized-LN token=${token} > logs/${model}_${data}_pretrained_features_size=${sent_num}_${token}.log 2>&1
    done
done

end=$(date +%s)
runtime=$((end - start))
echo "Running time: $runtime seconds"