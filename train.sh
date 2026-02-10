
export CUDA_VISIBLE_DEVICES=0

swift rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --model_type qwen2_5_vl \
  --dataset ./dataset/vqa_sft.jsonl \
  --train_type lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_steps 300 \
  --max_length 2048 \
  --loss_type dapo \
  --reward_funcs grpo_accuracy \
  --reward_weights 1 \
  --num_generations 4 \
  --temperature 1.0 \
  --save_steps 100 \
  --logging_steps 10 \
  --warmup_ratio 0.05 \
  --save_total_limit 4 \
  --output_dir ./outputs/qwen2_5vl_vqa_grpo \
  --bf16 false \
  --fp16 true \
  --gradient_checkpointing true \
  --use_hf

echo "Training complete!"
