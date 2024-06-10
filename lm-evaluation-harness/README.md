We used `lm-evaluation-harness` to evaluate Llma3 as it's officially used by [llama-recipes](https://github.com/meta-llama/llama-recipes/tree/main/recipes/evaluation)

We will make a request to merge our MMLU-SR branch to `lm-evaluation-harness` for convenience.
Right now, you need to Clone the lm-evaluation-harness repository and install it:
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

Once you have lm-evaluation-harness installed, you can copy and paste our lm-evaluation-harness folder to add the tasks of mmlusr into lm-eval

You can adjust arguments in launch.sh, to reproduce our experiment results, set --batch_size 2 and --num_fewshot 5.
change --tasks in launch.sh to run different dataset.
You can also test other models by changing --model and --model_args.
Details can be found in [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main)
Run the shell to start experiment, if not excutable, try:
```bash
chmod +x launch.sh
```
Then:
```bash
./launch.sh
```