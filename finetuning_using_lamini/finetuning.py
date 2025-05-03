import os
import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig  # PEFT freezes most weights; trains only selected layers (e.g., LoRA adapters)
from trl import SFTTrainer

# Model and dataset
model_name = "NousResearch/Llama-2-7b-chat-hf"
dataset_name = "mlabonne/guanaco-llama2-1k"
new_model = "Llama-2-7b-chat-finetune"

# LoRA/QLoRA Config
loran_r = 64
lora_alpha = 16
lora_dropout = 0.1

# Training Arguments
output_dir = "./results/"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-5
weight_decay = 0.001
optimizer = "paged_adamw_32bit"
scheduler = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 25

# SFT Trainer-specific
max_seq_length = None
packing = False

# Set device map (CPU only)
device_map = {"": 0}  # Adjust based on your system; use only CPU if no GPU is available.

# Load dataset
dataset = load_dataset(dataset_name, split="train")

# Load model without quantization (standard precision)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,  # Use CPU or available device
)
model.config.use_cache = False  # Disable cache for training
model.config.pretraining_tp = 1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA config
peft_config = LoraConfig(
    r=loran_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    task_type="CAUSAL_LM",
    bias="none",
)

# TrainingArguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optimizer,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=scheduler,
    report_to="tensorboard",
)

# Train with SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_arguments,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    packing=packing,
    dataset_text_field="text",
)

trainer.train()

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our trained model
prompt = "What is a large language model?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
