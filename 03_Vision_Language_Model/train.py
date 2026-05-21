import os
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, TrainingArguments, Trainer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model

# Import our custom dataset logic
from dataset import RFConstellationTrainDataset

# ====================================================================
# ENVIRONMENT & PATH CONFIGURATION
# ====================================================================
os.environ['HF_HOME'] = os.path.expanduser('~/hf_cache')
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Relative paths assuming data was generated in the adjacent folder
DATA_DIR = '../01_Dataset_Simulation'
OUTPUT_DIR = './VLM_Output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_IMG = os.path.join(DATA_DIR, 'train_images_production.npz')
TRAIN_LBL = os.path.join(DATA_DIR, 'train_labels_production.csv')
VAL_IMG = os.path.join(DATA_DIR, 'val_images_production.npz')
VAL_LBL = os.path.join(DATA_DIR, 'val_labels_production.csv')

required_files = [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]
if not all(os.path.exists(f) for f in required_files):
    raise FileNotFoundError("CRITICAL ERROR: Dataset files not found. Please run scripts in '01_Dataset_Simulation' first.")

# ====================================================================
# INITIALIZATION
# ====================================================================
print("Loading Processor and Base Model...")
processor = AutoProcessor.from_pretrained(MODEL_ID, max_image_pixels=384*384)

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
)

# Apply LoRA adapters to all linear layers (including Vision Encoder)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()

train_dataset = RFConstellationTrainDataset(TRAIN_IMG, TRAIN_LBL, split="train")
val_dataset = RFConstellationTrainDataset(VAL_IMG, VAL_LBL, split="val")

# ====================================================================
# COLLATOR & PADDING LOGIC
# ====================================================================
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

processor.tokenizer.padding_side = "right"

def custom_collate_fn(batch):
    images = [item["image"] for item in batch]
    messages = [item["messages"] for item in batch]

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = inputs["input_ids"].clone()

    # Mask padding tokens
    labels[inputs["attention_mask"] == 0] = -100

    # Mask user prompt so Loss is calculated ONLY on the assistant's answer
    im_start_token_id = 151644
    for i in range(len(messages)):
        token_ids = labels[i].tolist()
        if im_start_token_id in token_ids:
            # Mask everything before the LAST occurrence of <|im_start|>
            last_start_idx = len(token_ids) - 1 - token_ids[::-1].index(im_start_token_id)
            labels[i, :last_start_idx + 3] = -100

    inputs["labels"] = labels
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    return inputs

# ====================================================================
# TRAINING EXECUTION (HPC DDP ENABLED)
# ====================================================================
available_workers = min(8, max(1, (os.cpu_count() or 2) - 1))

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    ddp_find_unused_parameters=False,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    bf16=True,
    optim="adamw_torch_fused",
    gradient_checkpointing=True,
    dataloader_num_workers=available_workers,
    dataloader_pin_memory=True
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=custom_collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print(f"Starting PEFT Training on HPC... (Workers: {available_workers})")
trainer.train()

# Save final adapters
final_save_path = os.path.join(OUTPUT_DIR, "best_vlm_adapter")
trainer.save_model(final_save_path)
processor.save_pretrained(final_save_path)

print(f"Training Complete! Model saved to: {os.path.abspath(final_save_path)}")