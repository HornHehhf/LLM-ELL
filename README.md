# Equi-Learning Law
This is the code repository for the arXiv paper [A Law of Next-Token Prediction in Large Language Models](https://arxiv.org/pdf/2408.13442).
If you use this code for your work, please cite
```
@misc{he2024lawnexttokenpredictionlarge,
      title={A Law of Next-Token Prediction in Large Language Models}, 
      author={Hangfeng He and Weijie J. Su},
      year={2024},
      eprint={2408.13442},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.13442}, 
}
```
## Installing Dependencies
Use virtual environment tools (e.g. miniconda) to install packages and run experiments:\
conda create -n myenv python=3.10\
conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge\
pip3 install -r requirements.txt

## Code Organization
- analyzing_law.py (analyzing contextualized token embeddings)
- customized_pretraining_task.py (analyzing contextualized token embeddings with model-specific pretraining tasks)
- visualization.py (visualizing contextualized token embeddings)
- feature_learning.py (obtaining contextualized token embeddings)
- feature_quality_assessment.py (measures of contextualized token embeddings)
- utils.py (utility functions)
- run_*.sh (experiments scripts)

## Change the Dir Path
Change the /path/to/working/dir to your working directory

## Reproducing experiments

To reproduce the experiments
```
sh run_main_results.sh
sh run_visualization.sh
sh run_training_step.sh
sh run_model_scaling.sh
sh run_pretraining_task.sh
sh run_information_flow.sh
sh run_measure_analysis.sh
sh run_normalization_analysis.sh
sh run_probing_data_analysis.sh
```

## Using models released by [Scaling Data-Constrained Language Models](https://arxiv.org/pdf/2305.16264)
- Please skip this step if you don't want to use their models
- To use their models, please follow the instructions in [datablations](https://github.com/huggingface/datablations) to download and convert their models to the versions supported by huggingface/transformers
- Change the /path/to/model/dir to the path that you save their models

To reproduce the experiments with datablations models
```
sh run_training_epoch.sh
sh run_data_repetition.sh
sh run_pretraining_data_analysis.sh
```
