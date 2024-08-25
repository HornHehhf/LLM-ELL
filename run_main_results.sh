#!/bin/bash
start=$(date +%s)

declare -A probing_data
probing_data[openai-gpt]=bookcorpus
probing_data[gpt2-xl]=bookcorpus
probing_data[llama-13b]=oscar
probing_data[Llama-2-13b-hf]=pes2o
probing_data[Llama-2-13b-chat-hf]=pes2o
probing_data[Meta-Llama-3-8B]=bookcorpus
probing_data[Meta-Llama-3-8B-Instruct]=bookcorpus
probing_data[Mistral-7B-v0.1]=c4
probing_data[Mistral-7B-Instruct-v0.1]=c4
probing_data[Mistral-7B-v0.2]=c4
probing_data[Mistral-7B-Instruct-v0.2]=c4
probing_data[Mistral-7B-v0.3]=c4
probing_data[Mistral-7B-Instruct-v0.3]=c4
probing_data[phi-1_5]=bookcorpus
probing_data[phi-2]=bookcorpus
probing_data[Phi-3-medium-4k-instruct]=c4
probing_data[Phi-3-medium-128k-instruct]=c4
probing_data[rwkv-4-14b-pile]=c4
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

for model in openai-gpt gpt2-xl llama-13b Llama-2-13b-hf Llama-2-13b-chat-hf Meta-Llama-3-8B Meta-Llama-3-8B-Instruct Mistral-7B-v0.1 Mistral-7B-Instruct-v0.1 Mistral-7B-v0.2 Mistral-7B-Instruct-v0.2 Mistral-7B-v0.3 Mistral-7B-Instruct-v0.3 phi-1_5 phi-2 Phi-3-medium-4k-instruct Phi-3-medium-128k-instruct rwkv-4-14b-pile rwkv-raven-14b mamba-2.8b-hf
do
    data=${probing_data[${model}]}
    sent_num=${sent_nums[${data}]}
    batch_size=${batch_sizes[${data}]}
    echo "save ${model} ${data}"
    python analyzing_law.py model=${model} phase=pretrained data=${data} sent_num=${sent_num} batch_size=${batch_size} feature=save regression=prediction-residual normalize=initialized-LN > logs/${model}_${data}_pretrained_features_size=${sent_num}.log 2>&1
    echo "load ${model} ${data}"
    python analyzing_law.py model=${model} phase=pretrained data=${data} sent_num=${sent_num} batch_size=${batch_size} feature=load regression=prediction-residual normalize=initialized-LN > logs/${model}_${data}_pretrained_features_size=${sent_num}.log 2>&1
done

end=$(date +%s)
runtime=$((end - start))
echo "Running time: $runtime seconds"
