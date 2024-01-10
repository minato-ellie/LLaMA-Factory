CUDA_VISIBLE_DEVICES=0


# Base model and dataset

# ## japanese-stablelm-base-gamma-7b
# model_name_or_path=/mnt/models-1/base_models/japanese-stablelm-base-gamma-7b
# dataset=wikipedia_ja_2022,culturax_en,culturax_ja,wikipedia_en_2022

# ## zephyr-7b-beta
# model_name_or_path=/mnt/models-1/base_models/zephyr-7b-beta
# dataset=wikipedia_ja_2022,culturax_en,culturax_ja,wikipedia_en_2022

## new tokenizer
model_name_or_path=IbuNai/mistral-7b-ja-pt-fast-v0.2
dataset=wikipedia_ja_2022,culturax_en,culturax_ja,wikipedia_en_2022,github_code_all_mit
# dataset=wikipedia_ja_2022
mix_strategy=interleave_over
max_length=1024

# Training parameters
lora_rank=64
lora_alpha=128
lora_dropout=0.05
lora_target=q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj
additional_target=lm_head,embed_tokens,norm

output_dir=/content/outputs/pt/mistral_7b/adamw_10k/outputs
logging_dir=/content/outputs/pt/mistral_7b/adamw_10k/logs
report_to=all
num_train_epochs=1.0
max_steps=10000

per_device_train_batch_size=4
gradient_accumulation_steps=4
logging_steps=10
save_total_limit=100
save_steps=3000

learning_rate=5e-5
lr_scheduler_type=cosine
optim=adamw_torch
# warmup_steps=500
warmup_ratio=0.05
weight_decay=0.01

wandb_username="minato_ryan"
wandb_project="mistral"
# wandb_name="<some name>"
wandb_note="
max_length:${max_length},
lr:${learning_rate},
optim:${optim},
r:${lora_rank},
alpha:${lora_alpha},
dropout:${lora_dropout},
target:${lora_target},
trainable:${additional_target},
per_device_train_batch_size:${per_device_train_batch_size},
gradient_accumulation_steps:${gradient_accumulation_steps},
dataset:${dataset},
warmup_ratio:${warmup_ratio},
warmup_steps:${warmup_steps},
max_steps:${max_steps},
num_train_epochs:${num_train_epochs}
"


#     --overwrite_cache \
#     --overwrite_output_dir \
#    --additional_target ${additional_target} \

WANDB_ENTITY=${wandb_username} \
WANDB_PROJECT=${wandb_project} \
WANDB_NOTE=${wandb_note} \
nohup \
python src/train_bash.py \
    --stage pt \
    --model_name_or_path ${model_name_or_path} \
    --do_train \
    --dataset ${dataset} \
    --template mistral \
    --mix_strategy ${mix_strategy} \
    --max_length ${max_length} \
    \
    --finetuning_type lora \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --lora_target ${lora_target} \
    --additional_target ${additional_target} \
    \
    --output_dir ${output_dir} \
    --logging_dir ${logging_dir} \
    --report_to ${report_to} \
    --streaming \
    --max_steps ${max_steps} \
    --num_train_epochs ${num_train_epochs} \
    --logging_steps ${logging_steps} \
    --save_steps ${save_steps} \
    --save_total_limit ${save_total_limit} \
    \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --optim ${optim} \
    --learning_rate ${learning_rate} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --warmup_ratio ${warmup_ratio} \
    --weight_decay ${weight_decay} \
    \
    --plot_loss \
    --flash_attn \
    --fp16 \
> train_pt.log 2>&1 &
