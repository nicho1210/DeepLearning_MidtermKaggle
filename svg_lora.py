import torch
import os
import re
import time
import random
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch

from datasets import concatenate_datasets, load_dataset

print(torch.__version__)
print(torch.bfloat16)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Core training config.
# Keep runtime targets in line with contest_docs guidance (roughly <= 6-8 hours training).
"""
CONFIG = {
    "model_name": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",  # Verify exact ID from the linked Unsloth notebook.
    "max_seq_length": 2048,
    "lora_r": 16,
    "lora_alpha": 16,
    "learning_rate": 2e-4,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "logging_steps": 20,
    "eval_steps": 100,
    "save_steps": 200,
    "max_train_samples_per_source": 12000,
    "eval_size": 0.02,
    "output_dir": "/kaggle/working/qwen2b_svg_lora",
}
"""
CONFIG = {
    "model_name": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    "max_seq_length": 2048,
    "lora_r": 8,
    "lora_alpha": 16,
    "learning_rate": 1e-4,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "logging_steps": 20,
    "eval_steps": 100,
    "save_steps": 200,
    "max_train_samples_per_source": 12000,
    "eval_size": 0.02,
    "output_dir": "/content/qwen2b_svg_lora",
}
CONFIG

# Data catalog using the resources listed in contest_docs/03_Data_Design.md.
"""
DATASET_CATALOG = {
    "OmniSVG/MMSVG-Icon": {
        "split": "train",
        "prompt_fields": ["description", "keywords", "detail", "prompt", "text"],
        "svg_fields": ["svg", "picosvg", "completion", "target"],
    },

    "xingxm/SVGX-Core-250k": {
        "split": "train",
        "prompt_fields": ["qwen_caption", "img_analysis", "blip_caption", "name"],
        "svg_fields": ["svg_code"],
    },

    "xingxm/SVGX-SFT-1M": {
        "split": "train",
        "prompt_fields": ["prompt", "instruction", "input", "query"],
        "svg_fields": ["completion", "output", "svg", "response"],
    },
    "thesantatitan/deepseek-svg-dataset": {
        "split": "train",
        "prompt_fields": ["prompt", "instruction", "input"],
        "svg_fields": ["completion", "output", "svg"],
    },

    "local_train_csv": {
        "split": "train",
        "prompt_fields": ["prompt"],
        "svg_fields": ["svg"],  
    },
}
"""
DATASET_CATALOG = {
    "local_train_csv": {
        "split": "train",
        "prompt_fields": ["prompt"],
        "svg_fields": ["svg"],
    }
}

ACTIVE_SOURCES = [
    "local_train_csv",
]

print(DATASET_CATALOG)
print(ACTIVE_SOURCES)
# For a first run, keep to 1-2 sources.
ACTIVE_SOURCES = [
    #"xingxm/SVGX-Core-250k",
    #"xingxm/SVGX-SFT-1M",
    #"OmniSVG/MMSVG-Icon",
    "local_train_csv",
]

def _pick_first_non_empty(example, keys):
    for key in keys:
        if key in example and example[key] is not None:
            val = str(example[key]).strip()
            if val:
                return val
    return ""


def to_prompt_svg(example, prompt_fields, svg_fields):
    prompt = _pick_first_non_empty(example, prompt_fields)
    svg = _pick_first_non_empty(example, svg_fields)
    if not svg.lower().startswith("<svg"):
        return {"prompt": "", "svg": ""}
    return {"prompt": prompt, "svg": svg}

"""
def load_source_dataset(dataset_id, cfg, max_samples):
    print(f"Loading {dataset_id} ...")
    ds = load_dataset(dataset_id, split=cfg["split"])
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=SEED).select(range(max_samples))
    ds = ds.map(
        lambda ex: to_prompt_svg(ex, cfg["prompt_fields"], cfg["svg_fields"]),
        remove_columns=ds.column_names,
        desc=f"normalizing {dataset_id}",
    )
    ds = ds.filter(lambda x: bool(x["prompt"]) and bool(x["svg"]))
    print(f"{dataset_id}: {len(ds)} usable rows")
    return ds
"""
def load_source_dataset(dataset_id, cfg, max_samples):
    print(f"Loading {dataset_id} ...")

    if dataset_id == "local_train_csv":
        ds = load_dataset(
            "csv",
            data_files="train.csv",   # 如果不在同目錄，改成你的實際路徑
            split="train",
        )
    else:
        ds = load_dataset(dataset_id, split=cfg["split"])

    if max_samples and len(ds) > max_samples:
        #ds = ds.shuffle(seed=SEED).select(range(max_samples))
        ds = ds.select(range(max_samples))

    ds = ds.map(
        lambda ex: to_prompt_svg(ex, cfg["prompt_fields"], cfg["svg_fields"]),
        remove_columns=ds.column_names,
        desc=f"normalizing {dataset_id}",
    )

    ds = ds.filter(lambda x: bool(x["prompt"]) and bool(x["svg"]))
    print(f"{dataset_id}: {len(ds)} usable rows")
    return ds

"""
datasets_ok = []
for source in ACTIVE_SOURCES:
    try:
        ds = load_source_dataset(
            source,
            DATASET_CATALOG[source],
            CONFIG["max_train_samples_per_source"],
        )
        datasets_ok.append(ds)
    except Exception as e:
        print(f"Skipping {source}: {type(e).__name__}: {e}")

if not datasets_ok:
    raise RuntimeError("No dataset loaded. Check dataset IDs, internet access, and schema fields.")

train_raw = datasets_ok[0] if len(datasets_ok) == 1 else concatenate_datasets(datasets_ok)
train_raw = train_raw.shuffle(seed=SEED)

splits = train_raw.train_test_split(test_size=CONFIG["eval_size"], seed=SEED)
train_ds = splits["train"]
eval_ds = splits["test"]

print(f"Train rows: {len(train_ds)}")
print(f"Eval rows: {len(eval_ds)}")
train_ds[0]
"""
datasets_ok = []
for source in ACTIVE_SOURCES:
    try:
        ds = load_source_dataset(
            source,
            DATASET_CATALOG[source],
            CONFIG["max_train_samples_per_source"],
        )
        datasets_ok.append(ds)
    except Exception as e:
        print(f"Skipping {source}: {type(e).__name__}: {e}")

if not datasets_ok:
    raise RuntimeError("No dataset loaded. Check dataset IDs, internet access, and schema fields.")

train_raw = datasets_ok[0] if len(datasets_ok) == 1 else concatenate_datasets(datasets_ok)
train_raw = train_raw.shuffle(seed=SEED)

def keep_short_svg(x):
    svg = x["svg"]
    return (
        len(svg) <= 1200 and
        svg.count("<path") <= 8
    )

train_raw = train_raw.filter(keep_short_svg)

splits = train_raw.train_test_split(test_size=CONFIG["eval_size"], seed=SEED)
train_ds = splits["train"]
eval_ds = splits["test"]

print(f"Train rows: {len(train_ds)}")
print(f"Eval rows: {len(eval_ds)}")
print(train_ds[0])

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CONFIG["model_name"],
    max_seq_length=CONFIG["max_seq_length"],
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)


SYSTEM_PROMPT = (
    #"You generate compact, valid SVG markup from user requests. "
    #"Return only SVG code with a single root <svg> element."
    "You generate compact, valid SVG markup from user requests. "
    "Return only valid SVG markup. No explanations or extra text."
)

"""
def format_sft_text(example):
    text = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{example['prompt']}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{example['svg']}<|im_end|>"
    )
    return {"text": text}


train_text = train_ds.map(format_sft_text, remove_columns=train_ds.column_names)
eval_text = eval_ds.map(format_sft_text, remove_columns=eval_ds.column_names)
"""
def format_sft_text(example):
    text = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{example['prompt']}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{example['svg']}<|im_end|>"
    )
    return {"text": text}

train_text_all = train_ds.map(format_sft_text, remove_columns=train_ds.column_names)
eval_text_all = eval_ds.map(format_sft_text, remove_columns=eval_ds.column_names)

def keep_short_tokenized(example):
    n_tokens = len(tokenizer(example["text"], add_special_tokens=False)["input_ids"])
    return n_tokens <= 1800

train_text = train_text_all.filter(keep_short_tokenized)
eval_text = eval_text_all.filter(keep_short_tokenized)

print("Filtered train rows:", len(train_text))
print("Filtered eval rows:", len(eval_text))
print(train_text[0]["text"][:400])

from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_train_epochs"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    warmup_ratio=CONFIG["warmup_ratio"],
    weight_decay=CONFIG["weight_decay"],
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=CONFIG["logging_steps"],
    eval_strategy="steps",
    eval_steps=CONFIG["eval_steps"],
    save_steps=CONFIG["save_steps"],
    save_total_limit=2,
    report_to="none",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_text,
    eval_dataset=eval_text,
    dataset_text_field="text",
    max_seq_length=CONFIG["max_seq_length"],
    packing=False,
    args=training_args,
)

train_result = trainer.train()
train_result

os.makedirs(CONFIG["output_dir"], exist_ok=True)
trainer.save_model(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])

print(f"Saved adapter + tokenizer to: {CONFIG['output_dir']}")
print(os.listdir(CONFIG["output_dir"]))


from unsloth import FastLanguageModel
import torch, re, xml.etree.ElementTree as ET

# 重新載入 base model
infer_model, infer_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length=CONFIG["max_seq_length"],
    dtype=None,
    load_in_4bit=True,
)

# 載入 LoRA adapter
infer_model.load_adapter(CONFIG["output_dir"])

# Move to GPU
infer_model = infer_model.to("cuda")
infer_model.eval()

# FastLanguageModel.for_inference(infer_model)

SVG_REGEX = re.compile(r"<svg\b[^>]*>[\s\S]*?</svg>", flags=re.IGNORECASE)

def extract_svg(text):
    matches = SVG_REGEX.findall(text)
    return matches[-1].strip() if matches else ""

def is_valid_svg(svg_text):
    if not svg_text:
        return False
    try:
        root = ET.fromstring(svg_text)
        return root.tag.endswith("svg")
    except ET.ParseError:
        return False

def fallback_svg(_prompt):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">'
        '<rect x="0" y="0" width="256" height="256" fill="white"/>'
        '<circle cx="128" cy="128" r="64" fill="black"/>'
        '</svg>'
    )
"""
def generate_svg(prompt, max_new_tokens=256):
    chat_text = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = infer_tokenizer(
        chat_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")

    with torch.no_grad():
        output_ids = infer_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=infer_tokenizer.pad_token_id,
            eos_token_id=infer_tokenizer.eos_token_id,
            use_cache=False,   # ⭐ 先求穩
        )

    decoded = infer_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    svg = extract_svg(decoded)
    if not is_valid_svg(svg):
        svg = fallback_svg(prompt)
    return svg
"""
"""
def generate_svg(prompt, debug=False):
    chat_text = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = infer_tokenizer(
        chat_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")

    with torch.no_grad():
        output_ids = infer_model.generate(
            **inputs,
            #max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.15,
            pad_token_id=infer_tokenizer.pad_token_id,
            eos_token_id=infer_tokenizer.eos_token_id,
            use_cache=False,
        )

    decoded = infer_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if debug:
        print("==== RAW MODEL OUTPUT ====")
        print(decoded[:8000])
        print("==== END RAW OUTPUT ====")

    svg = extract_svg(decoded)

    if debug:
        print("==== EXTRACTED SVG ====")
        print(svg[:1000] if svg else "[EMPTY]")
        print("==== END EXTRACTED SVG ====")

    if not is_valid_svg(svg):
        if debug:
            print("Falling back to fallback_svg()")
        svg = fallback_svg(prompt)

    return svg
    """
def generate_svg(prompt, debug=False):
    chat_text = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = infer_tokenizer(
        chat_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")

    input_len = inputs["input_ids"].shape[1]
    max_new_tokens = max(128, min(2048, CONFIG["max_seq_length"] - input_len - 64))

    with torch.no_grad():
        output_ids = infer_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.15,
            pad_token_id=infer_tokenizer.pad_token_id,
            eos_token_id=infer_tokenizer.eos_token_id,
            use_cache=False,
        )

    decoded = infer_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if debug:
        print(decoded[:2000])

    svg = extract_svg(decoded)

    if not is_valid_svg(svg):
        svg = fallback_svg(prompt)

    return svg
"""
test_prompt = "The image features two orange squares with a microphone icon and an arrow connecting them, set against a white background."
pred_svg = generate_svg(test_prompt)
print(pred_svg[:500])
print("Valid SVG:", is_valid_svg(pred_svg))
"""
pred_svg = generate_svg(
    "The image features two orange squares with a microphone icon and an arrow connecting them, set against a white background.", 
    debug=True
)
print(pred_svg[:500])
print("Valid SVG:", is_valid_svg(pred_svg))
print("SVG length:", len(pred_svg))
print("Ends with </svg>:", pred_svg.strip().endswith("</svg>"))
print(pred_svg[-200:])

full_time_min = 1.52 / 5 * len(pd.read_csv("test.csv"))
print(full_time_min)

#-----------------------------Submission-------------------------------------
# Submission generation scaffold: expects Kaggle prompt file with columns `id,prompt`.
#TEST_PROMPTS_PATH = "/kaggle/input/svg-test-public-prompts/test_prompts.csv"
TEST_PROMPTS_PATH = "test.csv"
SUBMISSION_PATH = "submission.csv"

#test_df = pd.read_csv(TEST_PROMPTS_PATH)
test_df = pd.read_csv(TEST_PROMPTS_PATH)

rows = []
invalid_count = 0
t0 = time.time() 

for _, row in test_df.iterrows():
    #svg = generate_svg(row["prompt"])
    svg = generate_svg(row["prompt"], debug=False)
    if not is_valid_svg(svg):
        invalid_count += 1
        svg = fallback_svg(row["prompt"])
    rows.append({"id": row["id"], "svg": svg})

sub_df = pd.DataFrame(rows)
sub_df.to_csv(SUBMISSION_PATH, index=False)

elapsed_min = (time.time() - t0) / 60
print(f"Saved: {SUBMISSION_PATH}")
print(f"Rows: {len(sub_df)}")
print(f"Invalid/fallback count: {invalid_count}")
print(f"Runtime (minutes): {elapsed_min:.2f}")
sub_df.head()