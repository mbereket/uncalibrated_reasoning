## Biological Experiment Task (CRISPRi Perturb-seq)

Code for running the CRISPRi perturb-seq experiments

### View paper results only

`perturbation/view_results.ipynb`: load predictions and plot results. Uses paper predictions with `load_paper_data=True`

### Train models + generate your own predictions

`perturbation/train.sh`: train Qwen3-4B on biological experiment data (perturb-seq). May consider changing checkpoint frequency or removing unneeded checkpoints afterwards to save space.

`perturbation/run_merge.sh`: convert FSDP checkpoints to huggingface model checkpoint for each experiment

`perturbation/eval.sh`: generate predictions from checkpoints

`perturbation/view_results.ipynb`: load predictions and plot results. Uses your own predictions with `load_paper_data=False`
