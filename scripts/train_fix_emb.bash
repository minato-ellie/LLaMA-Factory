CUDA_VISIBLE_DEVICES=0
SEED=84


# Base model and dataset

# ## japanese-stablelm-base-gamma-7b
# model_name_or_path=/mnt/models-1/base_models/japanese-stablelm-base-gamma-7b
# dataset=wikipedia_ja_2022,culturax_en,culturax_ja,wikipedia_en_2022

# ## zephyr-7b-beta
# model_name_or_path=/mnt/models-1/base_models/zephyr-7b-beta
# dataset=wikipedia_ja_2022,culturax_en,culturax_ja,wikipedia_en_2022

## new tokenizer
# model_name_or_path=~/models/models-1/base_models/mistral-7b-japanese
model_name_or_path=IbuNai/mistral-7b-japanese
dataset=wikipedia_ja_2022,culturax_en,culturax_ja,wikipedia_en_2022,github_code_all_mit

# Huggingface Setting
hub_model_id=mistral-7b-japanese-pt-lora-step2
hub_strategy=checkpoint
push_to_hub=False
hub_private_repo=True

# Log Setting
report_to=wandb
WANDB_ENTITY=minato_ryan
WANDB_PROJECT=mistral-7b-japanese
WANDB_NOTES="Pre_train_Stage."

# Training parameters
mix_strategy=interleave_over
max_length=4096
name_module_trainable=lm_head,embed_tokens
checkpoint_dir=~/models/models-1/intermediate_models/mistral_7b/pt_2_adamw_cosine_12000_b16/outputs

output_dir=~/models/models-1/intermediate_models/mistral_7b/pt_2_adamw_cosine_36000_b32/outputs
logging_dir=~/models/models-1/intermediate_models/mistral_7b/pt_2_adamw_cosine_36000_b32/logs

max_steps=4000  # exp: 1000, product: 250000

# NOTE: batch=per_device_train_batch_size * gradient_accumulation_steps
per_device_train_batch_size=8 
gradient_accumulation_steps=4

# NOTE: name rule: pt_2_<optim>_<lr_scheduler_type>_<steps>_b<batch_size>
# exp: (Cartesian produc)
# - optim: adamw_torch (batchsize=16, lr=5e-5), lion_32bit (batchsize=16, lr=5e-5)
# - lr_scheduler_type: cosine, cosine_with_restarts, constant
# decision:
# - optim: 
# - lr_scheduler_type: 
optim=adamw_torch  # adamw_torch (default), lion_32bit, paged_adamw_32bit, lion_8bit, paged_lion_8bit
lr_scheduler_type=cosine  # cosine (default), cosine_with_restarts, constant, constant_with_warmup, 

warmup_steps=100  # default: 0, 20, 100
logging_steps=5
save_total_limit=5
save_steps=100
learning_rate=5e-5  # adamw_torch: 5e-5, lion: 5e-5
num_train_epochs=1.0

#     --overwrite_cache \
#     --overwrite_output_dir \

# python src/train_bash.py \
nohup python src/train_bash.py \
    --stage pt \
    --model_name_or_path ${model_name_or_path} \
    --do_train \
    \
    --dataset ${dataset} \
    --streaming \
    --template default \
    --mix_strategy ${mix_strategy} \
    --max_length ${max_length} \
    \
    --finetuning_type fix \
    --name_module_trainable ${name_module_trainable} \
    \
    --output_dir ${output_dir} \
    --logging_dir ${logging_dir} \
    --checkpoint_dir ${checkpoint_dir} \
    --hub_model_id ${hub_model_id} \
    --hub_private_repo ${hub_private_repo} \
    --hub_strategy ${hub_strategy} \
    --push_to_hub ${push_to_hub} \
    \
    --num_train_epochs ${num_train_epochs} \
    --max_steps ${max_steps} \
    --save_total_limit ${save_total_limit} \
    --save_steps ${save_steps} \
    --logging_steps ${logging_steps} \
    --report_to ${report_to} \
    \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    \
    --optim ${optim} \
    --learning_rate ${learning_rate} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --warmup_steps ${warmup_steps} \
    --plot_loss \
    --flash_attn \
    --bf16 \
    --seed ${SEED} \
> train_pt.log 2>&1 &
