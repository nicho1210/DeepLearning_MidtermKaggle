# DeepLearning_MidtermKaggle
Dataset
Required Files

Place the following files in the working directory:

train.csv
Columns: prompt, svg
test.csv
Columns: id, prompt

Data Preprocessing

The following filters are applied:

Validity Filtering
SVG must start with <svg>
Remove empty entries
Structural Filtering
SVG length ≤ 1600
<path> ≤ 10
<circle> ≤ 8
<rect> ≤ 8
Token Filtering
Token length ≤ 1700

Dataset Split
Train / Validation split = 50% / 50%

Model
Base model: unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
Parameters: ~0.9B
Quantization: 4-bit
LoRA Configuration
Rank: 8
Alpha: 16
Target modules:
q_proj, k_proj, v_proj, o_proj
gate_proj, up_proj, down_proj

🧠 Inference
Two-Stage Generation Strategy
First Pass (Stable)
temperature = 0.65
top_p = 0.90
max tokens ≈ 640
Second Pass (Fallback Attempt)
temperature = 0.80
top_p = 0.95
max tokens ≈ 2000

If both fail → fallback SVG is used.


✅ SVG Validation

Each SVG is checked using:

XML parsing (ElementTree)
Root tag must be <svg>

Invalid outputs → replaced with fallback SVG

⏱ Runtime
~671.6 minutes for 1000 samples
Sequential generation (no batching)
Each sample may use up to 2 passes

