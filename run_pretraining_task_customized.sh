#!/bin/bash
start=$(date +%s)

declare -A probing_data
probing_data[bert-base-uncased]=pes2o
probing_data[bert-large-uncased]=pes2o
probing_data[roberta-base]=pes2o
probing_data[roberta-large]=pes2o
probing_data[t5-base]=c4
probing_data[t5-large]=c4
probing_data[t5-3b]=c4
probing_data[t5-11b]=c4

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


for model in bert-base-uncased bert-large-uncased roberta-base roberta-large
do
    data=${probing_data[${model}]}
    sent_num=$((${sent_nums[${data}]} * 7))
    batch_size=${batch_sizes[${data}]}
    echo "save ${model} ${data}"
    python customized_pretraining_task.py model=${model} phase=pretrained data=${data} sent_num=${sent_num} batch_size=${batch_size} feature=save regression=prediction-residual normalize=initialized-LN > logs/${model}_${data}_pretrained_features_size=${sent_num}_mlm.log 2>&1
    echo "load ${model} ${data}"
    python customized_pretraining_task.py model=${model} phase=pretrained data=${data} sent_num=${sent_num} batch_size=${batch_size} feature=load regression=prediction-residual normalize=initialized-LN > logs/${model}_${data}_pretrained_features_size=${sent_num}_mlm.log 2>&1
done

for model in t5-base t5-large t5-3b t5-11b
do
    data=${probing_data[${model}]}
    sent_num=$((${sent_nums[${data}]} * 7))
    batch_size=${batch_sizes[${data}]}
    echo "save ${model} ${data}"
    python customized_pretraining_task.py model=${model} phase=pretrained data=${data} sent_num=${sent_num} batch_size=${batch_size} feature=save regression=prediction-residual normalize=initialized-LN > logs/${model}_${data}_pretrained_features_size=${sent_num}_sc.log 2>&1
    echo "load ${model} ${data}"
    python customized_pretraining_task.py model=${model} phase=pretrained data=${data} sent_num=${sent_num} batch_size=${batch_size} feature=load regression=prediction-residual normalize=initialized-LN > logs/${model}_${data}_pretrained_features_size=${sent_num}_sc.log 2>&1
done

end=$(date +%s)
runtime=$((end - start))
echo "Running time: $runtime seconds"
