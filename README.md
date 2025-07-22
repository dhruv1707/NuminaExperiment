# Tool-Augmented GRPO (Qwen2.5-3B-Instruct)

Post-trained **Qwen2.5-3B-Instruct** with a GRPO pipeline that adds a Python execution tool + new reward shaping to improve mathematical reasoning. After **1 epoch** the model scores **0.52 on MATH-500**, beating the larger **Qwen2.5-7B** (~0.50).

---

## Core Idea

Inject a *Python interpreter* into GRPO training so the policy can:
1. Think: `<think>...</think>`
2. Execute code: `<python>...</python>`
3. See runtime feedback: `<output>...</output>`
4. Produce final answer: `<answer>...</answer>`

Rewards encourage correct answers **and** reliable tool usage.

---

## Implementation

### Base Model
`Qwen2.5-3B-Instruct`.

### Reward Functions
Deterministic (parsed from the tagged output):
- **Format:** Output contains required tag sequence.
- **Accuracy:** 1 if final answer matches ground truth.
- **Tool Success:** `log(T_success / T_total)` — penalizes failed executions.

### Trainer Modifications
Using Hugging Face `GRPOTrainer` with custom hooks:
- Parse generated text for `<python>` blocks.
- Execute code in a sandbox; capture stdout/exception as `<output>`.
- Reinsert `<output>` into the model’s trajectory.
- Compute rewards and update policy (GRPO).

### Training
- Hardware: 1× A100 80GB
- Duration: 1 epoch (~3h)
- Data: Math reasoning problems; evaluation on **MATH-500**
- Standard HF optimizations (gradient checkpointing, etc.)

---

## Results

| Model | Params | Tool Aug? | Epochs | MATH-500 Accuracy |
|-------|--------|-----------|--------|-------------------|
| Qwen2.5-7B (baseline) | 7B | No | – | ~0.50 |
| **This Work (3B)** | 3B | Yes | 1 | **0.52** |

> Tool-augmented 3B model surpasses 7B baseline.

---

## Next
- More epochs
- Sub-1B experiments
- Release cleaned training/eval scripts