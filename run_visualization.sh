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
sent_nums[MedRAG-textbooks]=200
sent_nums[legalbench]=1000
sent_nums[us-congressional-speeches]=500

declare -A batch_sizes
batch_sizes[bookcorpus]=10
batch_sizes[c4]=1
batch_sizes[openwebtext]=1
batch_sizes[wiki]=1
batch_sizes[pes2o]=1
batch_sizes[pile]=1
batch_sizes[redpajama]=1
batch_sizes[oscar]=1
batch_sizes[MedRAG-textbooks]=1
batch_sizes[legalbench]=10
batch_sizes[us-congressional-speeches]=1


for model in openai-gpt
do
    for data in bookcorpus MedRAG-textbooks legalbench us-congressional-speeches
    do
        sent_num=${sent_nums[${data}]}
        batch_size=${batch_sizes[${data}]}
        echo "save ${model} ${data}"
        python analyzing_law.py model=${model} phase=pretrained data=${data} sent_num=${sent_num} batch_size=${batch_size} feature=save regression=prediction-residual normalize=initialized-LN > logs/${model}_${data}_pretrained_features_size=${sent_num}.log 2>&1
        echo "load ${model} ${data}"
        python analyzing_law.py model=${model} phase=pretrained data=${data} sent_num=${sent_num} batch_size=${batch_size} feature=load regression=prediction-residual normalize=initialized-LN > logs/${model}_${data}_pretrained_features_size=${sent_num}.log 2>&1
    done
done

end=$(date +%s)
runtime=$((end - start))
echo "Running time: $runtime seconds"


python visualization.py model=openai-gpt phase=pretrained data=MedRAG-textbooks sent_num=200 batch_size=1 feature=load regression=prediction-residual normalize=initialized-LN token_set=medicine
python visualization.py model=openai-gpt phase=pretrained data=legalbench sent_num=1000 batch_size=1 feature=load regression=prediction-residual normalize=initialized-LN token_set=law
python visualization.py model=openai-gpt phase=pretrained data=us-congressional-speeches sent_num=500 batch_size=1 feature=load regression=prediction-residual normalize=initialized-LN token_set=politics

python visualization.py model=openai-gpt phase=pretrained data=bookcorpus sent_num=3000 batch_size=1 feature=load regression=prediction-residual normalize=initialized-LN token_set=they-them
python visualization.py model=openai-gpt phase=pretrained data=bookcorpus sent_num=3000 batch_size=1 feature=load regression=prediction-residual normalize=initialized-LN token_set=have-had
python visualization.py model=openai-gpt phase=pretrained data=bookcorpus sent_num=3000 batch_size=1 feature=load regression=prediction-residual normalize=initialized-LN token_set=are-is
