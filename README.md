# MMLU-SR

This is the official repository for ["Reasoning or Simply Next Token Prediction? A Benchmark for Stress-Testing Large Language Models"](https://arxiv.org/abs/2406.15468).
- Our MMLU-SR dataset is designed to challenge true comprehension and reasoning abilities of LLMs by symbol replacement.
- Our evaluations on `gpt-3.5-turbo`, `gemini-1.0-pro`, and `llama3-8b` showed significantly lower performance on MMLU-SR compared to the original MMLU, highlighting the value of our MMLU-SR. 

![MMLU-SR](images/MMLU-SR2.PNG)

### Update History
+ [8/2024] Our MMLU-SR is evaluated on `gemini-1.5-pro`, `llama3-70b` and `gpt-4o-mini`, showing lower performance on MMLU-SR compared to the original MMLU. Check our [new experiment results](https://drive.google.com/file/d/1fsfEmBSxJIXcwQczFAsKsb0OqIKJKrrP/view)!
+ [7/2024] Our MMLU-SR is merged to [lm-eval-tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlusr)!
+ [6/2024] Project page set up at [Paper with Code](https://paperswithcode.com/dataset/mmlu-sr), with initial leaderboards for three MMLU-SR variants, [`Question Only`](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu-sr),  [`Answer Only`](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu-sr-1), and  [`Question and Answer`](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu-sr-2). 

## Environment Setup
It depends on what LLMs you are testing. To reproduce our experiment where we tested `gpt-3.5-turbo`, `gemini-1.0-pro` and `llama3-8b`. There is a `environment.yml` file as reference. First, you need to install conda in your system. Then you can run, assume running under Linux(we already found issues that Mac users will have package installation conflicts):
```bash
conda env create -f environment.yml
conda activate mmlusr
```
## Human evaluation form
We created human eval questionnaire which extracts 50 problems from MMLU/MMLU-SR. If you are intersted, please take some time to do it. We really appreciate for it :)
[human](https://docs.google.com/forms/d/e/1FAIpQLSdcvz2hMPC1YnOv1f3tOlyko0NL9ZfVc8SbSTFHDs6K77vHAg/viewform?usp=sf_link)

## Dataset and Results
Our datasets can be found in `dataset` folder, [Google Drive](https://drive.google.com/file/d/1ckqXmT7L2R0bWRccI60emZINkmFnTs6T/view?usp=drive_link), and also on [Huggingface](https://huggingface.co/datasets/NiniCat/MMLU-SR).
To evaluate our dataset using GPT and Gemini with specific task, you can run the following:
```bash
python3 evaluate.py --task question_only --model_name your_model
```
You can change task to `question_only`, `answer_only`, and `question_and_answer`.

Once you have the output json files, you can use `categories.py` to view the grouped results:
```bash
python3 categories.py 
```
For Llama3, you need to look into `lm-evaluation-harness` folder and follow the instruction.
The output of models' evaluation can be downloaded with this [Google Drive link](https://drive.google.com/file/d/1BKBx4LrkvU9WCnTREc1ENuRITT_xByI_/view?usp=sharing).
lm-evaluation-harness will automatically categorize all subjects results in its json output file.

## Huggingface 
We also provide a Hugging Face Dataset for users who want to use other frameworks like lm-evaluation-harness. 
To clone the entire dataset:
```bash
git clone https://huggingface.co/datasets/NiniCat/MMLU-SR
```

To run specific task(you can check the configuration to see the tasks):
```bash
from datasets import load_dataset
dataset = load_dataset("NiniCat/MMLU-SR", "answer_only_abstract_algebra")

```
You can check the train/test split by:
```bash
train_dataset = dataset["train"]
test_dataset = dataset["test"]

print(f"Number of training examples: {len(train_dataset)}")
print(f"Number of test examples: {len(test_dataset)}")
```
For running lm-eval below using huggingface models, you will need to first log in with your huggingface access token:
```bash
huggingface-cli login
```
## lm-eval
### Installation

Clone and install the `lm-evaluation-harness` as follows:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```
Run the desired groups as follows, for example, to reproduce experiment results from paper using LLaMA3 8B with MMLU-SR Question-and-Answer dataset:
```bash
lm_eval --model hf  --model_args pretrained=meta-llama/Meta-Llama-3-8B,parallelize=True  --tasks mmlusr   --batch_size 2  --output_path 'your path'
```
You can change the models simply by change the `model_args`, check `lm_eval -h` for argument help, and more instructions on [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness/tree/main).
You can also switch to evaluate other datasets or a single task, check below information:
#### Groups
- `mmlusr`: MMLU variant where the terminology in the question and answers are modified.
- `mmlusr_answer_only`: MMLU variant where the terminology in the answers are modified.
- `mmlusr_question_only`: MMLU variant where the terminology in the question is modified.
#### Tasks

There are 57 symbol replaced subjects in each group. You can run a single task by:

* `mmlusr_question_only_abstract_algebra`

Or by categories:

* `mmlusr_question_only_stem_tasks `

Or by subset groupts:

* `mmlusr_question_only`

## Experiment Results
Our experiments evaluated on `gpt-3.5-turbo`, `gemini-1.0-pro`,  `gemini-1.5-pro`, `llama3-8b` and `llama3-70b` are summarized in the table below:

| Model/Dataset          | Humanities | Social Sciences | STEM  | Other | Average | Avg Drop |
|------------------------|------------|-----------------|-------|-------|---------|----------|
| **GPT-3.5-turbo**      |            |                 |       |       |         |
| MMLU (5-shot)          | 0.723      | 0.770           | 0.554 | 0.714 | 0.677   |  -----   |
| Question Only (5-shot) | 0.661      | 0.702           | 0.506 | 0.641 | 0.616   |  9.08%   |
| Answer Only (5-shot)   | 0.540      | 0.595           | 0.441 | 0.538 | 0.520   | 23.27%   |
| Q&A (5-shot)           | 0.469      | 0.523           | 0.396 | 0.476 | 0.459   | 32.26%   |
| **Gemini-1.0-pro**     |            |                 |       |       |         |
| MMLU (5-shot)          | 0.728      | 0.758           | 0.596 | 0.703 | 0.686   |  -----   |
| Question Only (5-shot) | 0.687      | 0.744           | 0.539 | 0.658 | 0.645   |  5.86%   |
| Answer Only (5-shot)   | 0.619      | 0.670           | 0.504 | 0.591 | 0.586   | 14.48%   |
| Q&A (5-shot)           | 0.582      | 0.622           | 0.472 | 0.544 | 0.546   | 20.85%   |
| **Gemini-1.5-pro**     |            |                 |       |       |         |
| MMLU (5-shot)          | 0.849      | 0.881           | 0.802 | 0.815 | 0.832   |  -----   |
| Question Only (5-shot) | 0.795      | 0.836           | 0.700 | 0.754 | 0.764   |  8.17%   |
| Answer Only (5-shot)   | 0.741      | 0.816           | 0.747 | 0.739 | 0.758   |  8.89%   |
| Q&A (5-shot)           | 0.690      | 0.752           | 0.670 | 0.681 | 0.694   | 16.59%   |
| **Llama3-8B**          |            |                 |       |       |         |
| MMLU (5-shot)          | 0.593      | 0.757           | 0.557 | 0.729 | 0.651   |  -----   |
| Question Only (5-shot) | 0.546      | 0.685           | 0.507 | 0.668 | 0.595   |  8.69%   |
| Answer Only (5-shot)   | 0.455      | 0.599           | 0.460 | 0.557 | 0.510   | 21.28%   |
| Q&A (5-shot)           | 0.421      | 0.538           | 0.424 | 0.499 | 0.465   | 28.63%   |
| **Llama3-70B**         |            |                 |       |       |         |
| MMLU (5-shot)          | 0.681      | 0.868           | 0.697 | 0.814 | 0.765   |  -----   |
| Question Only (5-shot) | 0.635      | 0.812           | 0.631 | 0.770 | 0.712   |  6.93%   |
| Answer Only (5-shot)   | 0.539      | 0.683           | 0.565 | 0.622 | 0.602   | 21.31%   |
| Q&A (5-shot)           | 0.523      | 0.653           | 0.536 | 0.591 | 0.576   | 24.71%   |

## Citation
If you use this datasets in your work, please cite it as follows:
```bib
@misc{wang2024reasoning,
      title={Reasoning or Simply Next Token Prediction? A Benchmark for Stress-Testing Large Language Models}, 
      author={Wentian Wang and Paul Kantor and Jacob Feldman and Lazaros Gallos and Hao Wang},
      year={2024},
      eprint={2406.15468},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```
