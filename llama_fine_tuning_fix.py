
#%%capture
#%pip install accelerate peft bitsandbytes transformers trl


import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from accelerate import PartialState # FIX

# CUDA check available
print(torch.cuda.is_available())

# FIX 
torch.backends.cudnn.benchmark=True
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf" #"/home/soojung/llama-2-7b-chat-hf/"  # "NousResearch/Llama-2-7b-chat-hf"

# New instruction dataset
guanaco_dataset = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model
new_model = "llama-2-7b-chat-guanaco"

# Loading dataset, model, and tokenizer
dataset = load_dataset(guanaco_dataset, split="train")

# In our case, we create 4-bit quantization with NF4 type configuration using BitsAndBytes.
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

#  now load a model using 4-bit precision with the compute dtype "float16" from Hugging Face for faster training.
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map={"": PartialState().process_index}, # FIX 
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Loading Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Parameter-Efficient Fine-Tuning (PEFT)
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training parameters
"""
    output_dir: The output directory is where the model predictions and checkpoints will be stored.
    num_train_epochs: One training epoch.
    fp16/bf16: Disable fp16/bf16 training.
    per_device_train_batch_size: Batch size per GPU for training.
    per_device_eval_batch_size: Batch size per GPU for evaluation.
    gradient_accumulation_steps: This refers to the number of steps required to accumulate the gradients during the update process.
    gradient_checkpointing: Enabling gradient checkpointing.
    max_grad_norm: Gradient clipping.
    learning_rate: Initial learning rate.
    weight_decay: Weight decay is applied to all layers except bias/LayerNorm weights.
    Optim: Model optimizer (AdamW optimizer).
    lr_scheduler_type: Learning rate schedule.
    max_steps: Number of training steps.
    warmup_ratio: Ratio of steps for a linear warmup.
    group_by_length: This can significantly improve performance and accelerate the training process.
    save_steps: Save checkpoint every 25 update steps.
    logging_steps: Log every 25 update steps.
"""
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1, #4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# Model Fine-Tuning : Supervised Fine-Tuning (SFT)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# train
trainer.train()

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)



print("it is successful until now. Good job, Soojung :-)")
