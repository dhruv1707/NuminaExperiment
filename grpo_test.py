from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM
import re
from math_verify import LatexExtractionConfig, parse, verify
from trl import GRPOConfig
from trl import GRPOTrainer
from itertools import combinations

ds = load_dataset("AI-MO/NuminaMath-TIR")
train_dataset, test_dataset = ds['train'], ds['test']
print(train_dataset[0])

SYSTEM_PROMPT = (
    "You are a helpful assistant that can solve complex math problems step "
"by step with the help of a python executor tool . Given a question ,"
"you need to first think about the reasoning process in the mind "
"and then provide the answer . During thinking, you can write python "
"code , and invoke python tool to execute the code and get back the "
"output of the code . The reasoning process and answer are enclosed "
"within < think > </ think > and < answer > </ answer > tags respectively , "
"and the python code and the output are enclosed within < python > "
"</ python > and < output > </ output > tags respectively . You can "
"utilize the Sympy library to write the python code and make sure "
"to print the result at the end of the python code . You can utilize "
"the python tool as many times as required , however each python "
"code will be executed separately . For example , < think > reasoning "
"process here </ think > < python > python code here </ python > < output > "
"output of python code here </ output > < think > reasoning process "
"here </ think > < answer > final answer here </ answer >. "
)


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
        "solution": example["solution"]
    }


train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)

train_dataset = train_dataset.remove_columns(["messages", "problem"])
# print(train_dataset)

model_id = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    print("Checking format_reward function")
    tag_pairs = [["<think>", "</think>"], ["<python>", "</python>"], ["<output>", "</output>"], ["<answer>", "</answer>"]]
    pattern = r"^<think>.*?</think>\s*<python>.*?</python>\s*<output>.*?</output>\s*<answer>.*?</answer>$"
    rewards = []
    completion_contents = [completion[0]["content"] for completion in completions]
    for content in completion_contents:
        r = 0.0
        # print(f"Format reward function content: {content}")
        for pair in tag_pairs:
            if all(tag in content for tag in pair):
                if r < 0.5:
                    r += 0.125
        if re.match(pattern, content):
            r += 0.5
        rewards.append(r)

    # matches = [re.match(pattern, content) for content in completion_contents]
    # for match in matches:
    #     rewards[0] += 0.5 if match else 0.0
    print(f"Final answer from format_rewards: {rewards}, {len(rewards)}")
    return rewards

def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    print("Checking accuracy_reward")
    print(f"Completions shape: {len(completions)}")
    solutions = kwargs["solution"]
    print(f"Solutions shape: {len(solutions)}")
    rewards = []
    completion_contents = [completion[0]["content"] for completion in completions]
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        # print(f"kwargs: {kwargs}")
        # print(f"Content: {content}")
        # print(f"Solution: {solution}")
        if len(gold_parsed) != 0:
            try:
                if verify(answer_parsed, gold_parsed):
                    rewards.append(2.0)
                else:
                    rewards.append(0.0)
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    print(f"Final answer from accuracy_rewards: {rewards}, {len(rewards)}")
    return rewards

training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO-test",
    learning_rate=1e-6,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=False,
    temperature=1.0,
    # Parameters that control de data preprocessing
    max_completion_length=8000,  # default: 256
    num_generations=4,  # default: 8
    max_prompt_length=8,  # default: 512
    # Parameters related to reporting and saving
    report_to=["tensorboard"],
    logging_steps=10,
    push_to_hub=True,
    save_strategy="steps",
    save_steps=10,
)

trainer = GRPOTrainer(
    model=model, reward_funcs=[format_reward, accuracy_reward], args=training_args, train_dataset=train_dataset
)
print("Training starting")
trainer.train()

