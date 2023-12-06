CUDA_VISIBLE_DEVICES=0


# Base model and dataset

# ## japanese-stablelm-base-gamma-7b
# model_name_or_path=/mnt/models-1/base_models/japanese-stablelm-base-gamma-7b
# dataset=wikipedia_ja_2022,culturax_en,culturax_ja,wikipedia_en_2022

# ## zephyr-7b-beta
# model_name_or_path=/mnt/models-1/base_models/zephyr-7b-beta
# dataset=wikipedia_ja_2022,culturax_en,culturax_ja,wikipedia_en_2022

## new tokenizer
model_name_or_path=/mnt/models-1/base_models/mistral-7b-japanese
dataset=wikipedia_ja_2022,culturax_en,culturax_ja,wikipedia_en_2022,github_code_all_mit

# Training parameters
mix_strategy=interleave_over
max_length=4096
lora_rank=64
lora_alpha=128.0
lora_dropout=0.05
lora_target=q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj
additional_target=lm_head,embed_tokens
output_dir=/mnt/models-1/intermediate_models/mistral_7b/pt_2_outputs
logging_dir=/mnt/models-1/intermediate_models/mistral_7b/pt_2_logs
max_steps=250000
per_device_train_batch_size=4
gradient_accumulation_steps=4
lr_scheduler_type=cosine
warmup_steps=100
logging_steps=10
save_total_limit=8
save_steps=1000
learning_rate=2e-4
num_train_epochs=1.0

nohup python src/train_bash.py \
    --stage pt \
    --model_name_or_path ${model_name_or_path} \
    --do_train \
    --do_eval \
    --dataset ${dataset} \
    --template default \
    --mix_strategy ${mix_strategy} \
    --max_length ${max_length} \
    --finetuning_type lora \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --lora_target ${lora_target} \
    --additional_target ${additional_target} \
    --output_dir ${output_dir} \
    --logging_dir ${logging_dir} \
    --overwrite_cache \
    --overwrite_output_dir \
    --streaming \
    --max_steps ${max_steps} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --warmup_steps ${warmup_steps} \
    --logging_steps ${logging_steps} \
    --save_steps ${save_steps} \
    --save_total_limit ${save_total_limit} \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${num_train_epochs} \
    --plot_loss \
    --flash_attn \
    --bf16 \
> train_pt.log 2>&1 &
