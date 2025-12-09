from trl import SFTConfig, SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset, Dataset
from peft import LoraConfig
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from no_grad.patch.transformers import apply_patch

MODEL_NAME = "R1-Distill-Llama-8B-Hard-r1024-MoT"
MODEL = "qwen/Qwen3-0.6B-Base"
DATASET = "open-r1/Mixture-of-Thoughts"
MAX_STEPS = int(os.getenv("MAX_STEPS", "1")) #16_000
MAX_LENGTH = 17_000
LORA_RANK = 1024
LORA_ALPHA = 1024
USE_RSLORA = True
DEEPSPEED = None
LOGGING_STEPS = 1
SAVE_STEPS = 500
LR = float(os.getenv("LR", "2e-5"))
SCHEDULER_TYPE = "cosine_with_min_lr"
SCHEDULER_KWARGS = {"min_lr_rate": 0.05}
MAX_GRAD_NORM = 0.2
WARMUP_STEPS = 0
WARMUP_RATIO = 0.05
REPORT_TO_WANDB = bool(int(os.getenv("REPORT_TO_WANDB", "0")))
PUSH_TO_HUB = False
GRAD_ACCUM_STEPS = int(os.getenv("ACCUM_STEPS", "1"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16")) # original batch size was 128
EPOCHS = 1
USE_ES = bool(int(os.getenv("USE_ES", "0")))
ES_ARGS = {
    "population_size": int(os.getenv("ES_POPULATION_SIZE", "8")),
    "step_size": float(os.getenv("ES_STEP_SIZE", "2e-5")),
}
OPTIMIZER = os.getenv("OPTIMIZER", "adamw_torch")  # "sgd"
ADAM_BETAS = [float(b) for b in os.getenv("ADAM_BETAS", "0.9,0.999").split(",")]
if USE_ES:
    if OPTIMIZER == "adamw_torch":
        MODEL_NAME = f"R1-Distill-Llama-8B-Hard-r1024-MoT-lr{LR}-s{ES_ARGS['step_size']}-b{GRAD_ACCUM_STEPS * BATCH_SIZE}-p{ES_ARGS['population_size']}-B{ADAM_BETAS[0]}_{ADAM_BETAS[1]}-es_adam-vb_sweep-base"
    else:
        MODEL_NAME = f"R1-Distill-Llama-8B-Hard-r1024-MoT-lr{LR}-s{ES_ARGS['step_size']}-b{GRAD_ACCUM_STEPS * BATCH_SIZE}-p{ES_ARGS['population_size']}-no_adam-vb_sweep-base"
else:
    MODEL_NAME = f"R1-Distill-Llama-8B-Hard-r1024-MoT-lr{LR}-b{GRAD_ACCUM_STEPS * BATCH_SIZE}-B{ADAM_BETAS[0]}_{ADAM_BETAS[1]}-sgd_adam-vb_sweep-base"
RUN_ID = MODEL_NAME
DO_SAVE = False

if USE_ES:
    apply_patch()


dataset = load_dataset(DATASET, "all",
                       streaming=True)["train"].take(MAX_STEPS * GRAD_ACCUM_STEPS * BATCH_SIZE)


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
    save_strategy="steps" if DO_SAVE else "no",
    overwrite_output_dir=True,
    use_liger_kernel=True,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LR,
    lr_scheduler_type=SCHEDULER_TYPE,
    lr_scheduler_kwargs=SCHEDULER_KWARGS,
    max_grad_norm=MAX_GRAD_NORM,
    warmup_steps=WARMUP_STEPS,
    warmup_ratio=WARMUP_RATIO,
    max_steps=MAX_STEPS * EPOCHS,
    bf16=True,
    ddp_timeout=180000000,
    push_to_hub=PUSH_TO_HUB,
    hub_model_id=MODEL_NAME,
    hub_revision=RUN_ID,
    hub_private_repo=True,
    num_train_epochs=EPOCHS,
    optim=OPTIMIZER,
    adam_beta1=ADAM_BETAS[0],
    adam_beta2=ADAM_BETAS[1],
)

training_args.use_es = USE_ES
training_args.es_args = ES_ARGS

trainer = SFTTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()
