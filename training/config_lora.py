"""
LoRA Configuration for TinyLlama fine-tuning.
Tuned for 6 GB VRAM (e.g. RTX 3060 / 4060).
"""
from peft import LoraConfig, TaskType

def get_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                            # Reduced rank → less adapter memory
        lora_alpha=16,                  # Scaling factor (2 × r)
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )

# Training hyperparameters (used by fine_tune.py)
TRAIN_CONFIG = {
    "learning_rate":    2e-4,
    "per_device_train_batch_size": 1,   # 6 GB GPU can only hold 1 sample at a time
    "gradient_accumulation_steps": 16,  # Effective batch = 1 × 16 = 16
    "num_train_epochs": 3,
    "weight_decay":     0.01,
    "warmup_steps":     100,
    "lr_scheduler_type": "cosine",
    "logging_steps":    50,
    "save_steps":       200,
    "eval_steps":       200,
    "eval_strategy": "steps",
    "fp16":             True,
    "output_dir":       "training/fine_tuned_model",
    "report_to":        "none",
}

