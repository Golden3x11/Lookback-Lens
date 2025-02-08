#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=1G
#SBATCH --time=10:10:00
#SBATCH --account=plghallugrant-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --gres=gpu:1

conda activate myenv

srun python step04_run_decoding_our.py \
    --auth_token <TOKEN> \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --output_path nq_df_lookback_clf_cnndm-decoding \
    --num_gpus 1 \
    --do_sample \
    --guiding_classifier classifiers/classifier_anno-cnndm-7b_sliding_window_8.pkl \
    --chunk_size 8 \
    --num_candidates 8 \
    --max_new_tokens 200 \
    --start 1500 \
    --end 2000 \
    --temperature 0.7 \
    --top_p 0.7 \
    --task_type qa \
    --ds_name nq_df.parquet

