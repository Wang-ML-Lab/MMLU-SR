"Code originally copied from gemini-benchmark https://github.com/neulab/gemini-benchmark/blob/main/benchmarking/MMLU/run_mmlu.py"

from litellm import acompletion
from tqdm import tqdm
from utils import *
import pandas as pd
import asyncio
import litellm
import json

# parse arguments
import argparse
import os
import time


async def get_response(prompt: str, model: str):
    if "gemini" in model:
        response = await acompletion(
            model=model,
            messages=[
                {
                    "role": "system",
                    # To perform system instruction, add this prompt: In each of the questions that I ask, I will replace some of the words that you might know with a word that is arbitrarily assigned a specific meaning just for this test. The meaning of these arbitrary definition may change with every question. 
                    "content": "Follow the given examples and answer the question. Please respond to each question with 'Answer: <letter>' where <letter> is the correct choice. Avoid additional explanations.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE", #uses "BLOCK_ONLY_HIGH" if you don't have permission to "BLOCK_NONE".
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ],
        )
    else:
        response = await acompletion(
            model=model,
            messages=[
                {
                    "role": "system",
                    # To perform system instruction, add this prompt: In each of the questions that I ask, I will replace some of the words that you might know with a word that is arbitrarily assigned a specific meaning just for this test. The meaning of these arbitrary definition may change with every question. 
                    "content": "Follow the given examples and answer the question. Please respond to each question with 'Answer: <letter>' where <letter> is the correct choice. Avoid additional explanations.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
    return response


def main(args, tasks=TASKS):
    if "gpt" in args.model_name:
        # gpt evaluation
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    elif "gemini" in args.model_name:
        # gemini evaluation
        litellm.vertex_project = " "  # Your Project ID
        litellm.vertex_location = " "  # Your Project Location, e.g us-east4
        litellm.drop_params = True

    all_acc = 0
    all_number = 0
    accs_json = {}
    method_name = "5-shot" # We used 5-shot as baseline experiment
    outputs_file = open(f"results/{args.task}_{args.model_name}_{method_name}_outputs.json", "a")
    for task in tasks:
        print("Testing %s ..." % task)
        acc = 0
        dev_df = pd.read_csv(
            os.path.join("dataset/" + args.task + "_dev", args.task + "_" + task + "_dev.csv"), header=None
        )[: args.num_examples]
        test_df = pd.read_csv(
            os.path.join("dataset/" + args.task + "_test", args.task + "_" + task + "_test.csv"), header=None
        )
        for i in tqdm(range(test_df.shape[0])):
            try:
                # Building the prompt
                k = args.num_examples
                prompt_end = format_example(test_df, i, include_answer=False)
                train_prompt = gen_prompt(dev_df, task, k)
                prompt = train_prompt + prompt_end
                label = test_df.iloc[i, test_df.shape[1] - 1]

                # Get the model response
                response = asyncio.run(get_response(prompt, args.model_name))
                response_text = response["choices"][0]["message"]["content"]
                choices = {
                    "A": test_df.iloc[i, 1],
                    "B": test_df.iloc[i, 2],
                    "C": test_df.iloc[i, 3],
                    "D": test_df.iloc[i, 4]
                }
                ans_model = extract_ans(response_text, choices)

                correct = ans_model == label
                if correct:
                    acc += 1

                # Write the output to a file
                outputs_file.write(
                    json.dumps(
                        {
                            "task": task,
                            "correct": correct,
                            "prediction": ans_model,
                            "label": label,
                            "response": response_text,
                            "question": test_df.iloc[i, 0],
                            "A": test_df.iloc[i, 1],
                            "B": test_df.iloc[i, 2],
                            "C": test_df.iloc[i, 3],
                            "D": test_df.iloc[i, 4],
                            "prompt": prompt,
                        }
                    ) + "\n"
                )

            except Exception as e:
                print(f"Skipping index {i} due to error: {e}")
                continue  

        print("%s acc %.4f" % (task, acc / test_df.shape[0]))
        accs_json[task] = acc / test_df.shape[0]
        all_acc += acc
        all_number += test_df.shape[0]
    accs_json["all"] = all_acc / all_number
    json.dump(
        accs_json, open(f"results/{args.task}_{args.model_name}_{method_name}_accs.json", "w")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-1.0-pro",
        choices=["gpt-3.5-turbo", "gpt-4-1106-preview", "gemini-1.0-pro", "mixtral"],
    )
    parser.add_argument("--task", type=str, required=True, help="'answer_only', 'question_only', 'question_and_answer'")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of examples included in the current prompt input. ",
    )
    args = parser.parse_args()
    main(args)