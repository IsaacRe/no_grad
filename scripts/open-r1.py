from trl import SFTConfig, SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset, Dataset
from peft import LoraConfig
import torch
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    model_name: str = "R1-Distill-Llama-8B-Hard-r1024-MoT"
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    dataset: str = "open-r1/Mixture-of-Thoughts"
    max_steps: int = 16_000
    max_length: int = 17_000
    lora_rank: int = 1024
    lora_alpha: int = 1024
    use_rslora: bool = True
    deepspeed: str | None = None
    logging_steps: int = 1
    save_steps: int = 500
    lr: float = 2e-5
    scheduler_type: str = "cosine_with_min_lr"
    scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr_rate": 0.05})
    max_grad_norm: float = 0.2
    warmup_steps: int = 0
    warmup_ratio: float = 0.05
    report_to_wandb: bool = True
    push_to_hub: bool = False
    grad_accum_steps: int = 64
    batch_size: int = 2
    epochs: int = 3


MODEL_NAME = "R1-Distill-Llama-8B-Hard-r1024-MoT"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DATASET = "open-r1/Mixture-of-Thoughts"
MAX_STEPS = 16_000
MAX_LENGTH = 17_000
LORA_RANK = 1024
LORA_ALPHA = 1024
USE_RSLORA = True
DEEPSPEED = None
LOGGING_STEPS = 1
SAVE_STEPS = 500
LR = 2e-5
SCHEDULER_TYPE = "cosine_with_min_lr"
SCHEDULER_KWARGS = {"min_lr_rate": 0.05}
MAX_GRAD_NORM = 0.2
WARMUP_STEPS = 0
WARMUP_RATIO = 0.05
REPORT_TO_WANDB = True
PUSH_TO_HUB = False
GRAD_ACCUM_STEPS = 64
BATCH_SIZE = 2
EPOCHS = 3


dataset = load_dataset(DATASET, "all",
                       streaming=True)["train"].take(MAX_STEPS * GRAD_ACCUM_STEPS)


# SHAREGPT_ROLE_MAP = {
#     "human": "user",
#     "gpt": "assistant",
# }

# def fix_format(sample):
#     sample['messages'] = [
#         {
#             "role": SHAREGPT_ROLE_MAP[m["from"]],
#             "content": m["value"],
#         }
#         for m in sample["conversations"]
#     ]
#     return sample
# dataset = dataset.map(fix_format)

from functools import partial

def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds

dataset = Dataset.from_generator(partial(gen_from_iterable_dataset, dataset), features=dataset.features)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# The model to optimise
model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)

peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.0,
    use_rslora=USE_RSLORA,
)

training_args = SFTConfig(
    output_dir=f"{MODEL_NAME}/{RUN_ID}",
    per_device_train_batch_size=BATCH_SIZE,
    report_to="wandb" if REPORT_TO_WANDB else "none",
    run_name=MODEL_NAME,
    max_length=MAX_LENGTH,
    deepspeed=DEEPSPEED,
    dataloader_num_workers=16,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    overwrite_output_dir=True,
    use_liger_kernel=True,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LR,
    lr_scheduler_type=SCHEDULER_TYPE,
    lr_scheduler_kwargs=SCHEDULER_KWARGS,
    max_grad_norm=MAX_GRAD_NORM,
    warmup_steps=WARMUP_STEPS,
    warmup_ratio=WARMUP_RATIO,
    max_steps=MAX_STEPS,
    bf16=True,
    ddp_timeout=180000000,
    push_to_hub=PUSH_TO_HUB,
    hub_model_id=MODEL_NAME,
    hub_revision=RUN_ID,
    hub_private_repo=True,
    num_train_epochs=EPOCHS,
)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()
