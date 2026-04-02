import os
import re
import time
import random
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch

from datasets import concatenate_datasets, load_dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

CONFIG = {
    "model_name": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    #"model_name": "unsloth/Qwen2.5-4B-Instruct-bnb-4bit",
    "max_seq_length": 2048,
    "lora_r": 8,
    "lora_alpha": 16,
    "learning_rate": 1e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "logging_steps": 20,
    "eval_steps": 100,
    "save_steps": 200,
    "max_train_samples_per_source": 50000,
    "eval_size": 0.5,
    "output_dir": "/content/qwen2b_svg_lora",
}
CONFIG

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

def load_source_dataset(dataset_id, cfg, max_samples):
    print(f"Loading {dataset_id} ...")

    if dataset_id == "local_train_csv":
        ds = load_dataset(
            "csv",
            data_files="train.csv",   
            split="train",
        )
    else:
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
        len(svg) <= 1600 and
        svg.count("<path") <= 10 and
        svg.count("<circle") <= 8 and
        svg.count("<rect") <= 8
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

total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params:,}")
print(f"In billions: {total_params / 1e9:.2f}B")

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
    return n_tokens <= 1700

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
import torch
import re
import xml.etree.ElementTree as ET

infer_model, infer_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length=CONFIG["max_seq_length"],
    dtype=None,
    load_in_4bit=True,
)

# LoRA adapter
infer_model.load_adapter(CONFIG["output_dir"])

# Move to GPU
infer_model = infer_model.to("cuda")
infer_model.eval()

SVG_REGEX = re.compile(r"<svg\b[^>]*>[\s\S]*?</svg>", flags=re.IGNORECASE)

def extract_svg(text: str) -> str:
    matches = SVG_REGEX.findall(text)
    return matches[-1].strip() if matches else ""

def is_valid_svg(svg_text: str) -> bool:
    if not isinstance(svg_text, str) or not svg_text.strip():
        return False
    try:
        root = ET.fromstring(svg_text)
        if "}" in root.tag:
            tag = root.tag.split("}", 1)[1]
        else:
            tag = root.tag
        return tag == "svg"
    except ET.ParseError:
        return False

def fallback_svg(_prompt: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">'
        '<rect x="0" y="0" width="256" height="256" fill="white"/>'
        '<circle cx="128" cy="128" r="64" fill="black"/>'
        '</svg>'
    )

def postprocess_svg(svg: str) -> str:
    if not isinstance(svg, str):
        return ""
    svg = re.sub(r'\bfilling\s*=\s*"([^"]*)"', r'fill-rule="\1"', svg, flags=re.IGNORECASE)
    return svg.strip()

def generate_svg_once(
    prompt: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_cap: int = 768,
    debug: bool = False,
) -> str:
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
    max_new_tokens = max(128, min(max_cap, CONFIG["max_seq_length"] - input_len - 64))

    with torch.no_grad():
        output_ids = infer_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=infer_tokenizer.pad_token_id,
            eos_token_id=infer_tokenizer.eos_token_id,
            use_cache=False,
        )

    decoded = infer_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if debug:
        print("==== RAW MODEL OUTPUT ====")
        print(decoded[:3000])
        print("==== END RAW MODEL OUTPUT ====")

    svg = extract_svg(decoded)
    svg = postprocess_svg(svg)

    if debug:
        print("==== EXTRACTED SVG ====")
        print(svg[:1000] if svg else "[EMPTY]")
        print("==== END EXTRACTED SVG ====")

    return svg

def generate_svg(prompt: str, debug: bool = False):

    svg = generate_svg_once(
        prompt,
        temperature=0.65,
        top_p=0.90,
        repetition_penalty=1.08,
        max_cap=640,
        debug=debug,
    )
    if is_valid_svg(svg):
        return svg, False

    svg = generate_svg_once(
        prompt,
        temperature=0.80,
        top_p=0.95,
        repetition_penalty=1.12,
        max_cap=2000,
        debug=debug,
    )
    if is_valid_svg(svg):
        return svg, False

    return fallback_svg(prompt), True

test_prompt = "The image features two orange squares with a microphone icon and an arrow connecting them, set against a white background."
pred_svg, used_fallback = generate_svg(test_prompt, debug=True)

print(pred_svg[:500])
print("Valid SVG:", is_valid_svg(pred_svg))
print("Used fallback:", used_fallback)
print("SVG length:", len(pred_svg))
print("Ends with </svg>:", pred_svg.strip().endswith("</svg>"))
print(pred_svg[-200:])

TEST_PROMPTS_PATH = "test.csv"
SUBMISSION_PATH = "submission.csv"

test_df = pd.read_csv(TEST_PROMPTS_PATH)

rows = []
invalid_count = 0
t0 = time.time()
count = 0

for _, row in test_df.iterrows():
    count += 1
    print(count)
    svg, used_fallback = generate_svg(row["prompt"], debug=False)
    if used_fallback:
        invalid_count += 1
    rows.append({"id": row["id"], "svg": svg})

sub_df = pd.DataFrame(rows)
sub_df.to_csv(SUBMISSION_PATH, index=False)

elapsed_min = (time.time() - t0) / 60
print(f"Saved: {SUBMISSION_PATH}")
print(f"Rows: {len(sub_df)}")
print(f"Invalid/fallback count: {invalid_count}")
print(f"Runtime (minutes): {elapsed_min:.2f}")
sub_df.head()