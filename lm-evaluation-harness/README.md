# Evaluating Llama3 with lm-evaluation-harness

We have utilized the `lm-evaluation-harness`, officially supported by [llama-recipes](https://github.com/meta-llama/llama-recipes/tree/main/recipes/evaluation), to evaluate our Llama3 models. Our MMLU-SR has been merged to lm-eval already, so you can directly follow main page to call lm-eval to run. Here is instruction for local run.

## Getting Started

### Installation

Clone and install the `lm-evaluation-harness` as follows:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

## Configure Tasks
Once installed, integrate the MMLU-SR tasks into the `lm-evaluation-harness` by copying our task folder into your local `lm-evaluation-harness` installation. You can do this with the following command (modify the source path as needed):
```bash
cp -r /your path/lm-eval/tasks/mmlusr lm-evaluation-harness/lm_eval/tasks
```

### Adjust Experiment Settings

Modify the `launch.sh` script to tailor the experiment to your needs:

- **Batch Size**: Set `--batch_size 2` to specify the number of samples processed at once.
- **Few-Shot Setting**: Use `--num_fewshot 5` to define the number of examples used for few-shot learning.
- **Tasks**: Change `--tasks` in `launch.sh` to specify which dataset to run. To run specific single subject, e.g, `question_only_prehistory`. To run the group dataset. e.g, `mmlusr_question_only`.
- **Model Configuration**: Alter `--model` and `--model_args` to test different models.


Find more details in the official [lm-evaluation-harness documentation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) and [lm-eval new task](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md)

## Running the Experiment
Ensure the `launch.sh` script is executable. If it's not, make it executable with:
```bash
chmod +x launch.sh
```
Then, start the experiment:
```bash
./launch.sh
```
