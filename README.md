##  OVOR: OnePrompt with Virtual Outlier Regularization for Rehearsal-Free Class-Incremental Learning
PyTorch code for the ICLR 2024 paper:\
**OVOR: OnePrompt with Virtual Outlier Regularization for Rehearsal-Free Class-Incremental Learning**\
*Wei-Cheng Huang, Chun-Fu (Richard) Chen, Hsiang Hsu* \
The Twelfth International Conference on Learning Representations (ICLR), 2024

## Abstract

Recent works have shown that by using large pre-trained models along with learnable prompts, rehearsal-free methods for class-incremental learning (CIL) settings can achieve superior performance to prominent rehearsal-based ones. Rehearsal-free CIL methods struggle with distinguishing classes from different tasks, as those are not trained together. In this work we propose a regularization method based on virtual outliers to tighten decision boundaries of the classifier, such that confusion of classes among different tasks is mitigated. Recent prompt-based methods often require a pool of task-specific prompts, in order to prevent overwriting knowledge of previous tasks with that of the new task, leading to extra computation in querying and composing an appropriate prompt from the pool. This additional cost can be eliminated, without sacrificing accuracy, as we reveal in the paper. We illustrate that a simplified prompt-based method can achieve results comparable to previous state-of-the-art (SOTA) methods equipped with a prompt pool, using much less learnable parameters and lower inference cost. Our regularization method has demonstrated its compatibility with different promptbased methods, boosting those previous SOTA rehearsal-free CIL methods’ accuracy on the ImageNet-R and CIFAR-100 benchmarks.

## Getting Started

With Python version 3.8, run pip install to install the dependencies: `pip install -r requirements.txt`

### Datasets

Before running any scripts, create a folder `data/` at root, then unpack the corresponding datasets under this folder:
* CIFAR-100: could be automatically downloaded and extracted
* ImageNet-R: get it from https://github.com/hendrycks/imagenet-r
* ImageNet-A: get it from https://github.com/hendrycks/natural-adv-examples
* CUB-200: get it from https://www.vision.caltech.edu/datasets/cub_200_2011/

### Training
All commands should be run under the project root directory. **The scripts are set up for 4 GPUs** but can be modified for your hardware.
By default they run the same experiments for 5 times with different random seeds, with the number of times being adjustable in the scripts.
```bash
sh experiments/cifar100.sh
sh experiments/imagenet-r.sh
```
For 5/20-task ImageNet-R, ImageNet-A and CUB-200, use the corresponding yaml files (`imnet-r_prompt_short.yaml`, `imnet-r_prompt_long.yaml`, `imnet-a_prompt.yaml`, `cub200_prompt.yaml`) under the `configs` folder, in a similar way as `experiments/cifar100.sh` and `experiments/imagenet-r.sh`.

For hyperparameters, options common to all datasets are specified in `configs/common.yaml`, then options specified in dataset-specific yaml files (provided via `--config` command-line options) are loaded, and finally those hyperparameters specified via command-line options.
For example, executing
```bash
python -u run.py --config configs/imnet-r_prompt.yaml --prompt_param 10 40 10
```
runs OnePrompt, without virtual outlier regularization, on 10-task ImageNet-R.
By default it runs 50 epochs of unregularized training, by providing command-line arguments as follows:
```bash
python -u run.py --config configs/imnet-r_prompt.yaml --prompt_param 10 40 10 --epochs 0 40 10
```
It runs 40 epochs of unregularized training, followed by 10 epochs of regularized training.
The regularization is orthogonal to prompting method, e.g. CODA-Prompt with regularization:
```bash
python -u run.py --config configs/imnet-r_prompt.yaml --prompt_param 10 40 10 --epochs 0 40 10 --learner_name CODAPrompt
```
The similar goes for DualPrompt and L2P.
Please check out `experiments/imagenet-r.sh` for more explanations about `prompt_param` option.

Results will be written into `output.log` under the given `log_dir` folder, by default logs and checkpoints are saved under the `outputs` folder.

## Acknowledgement

Our code is adapted from https://github.com/GT-RIPL/CODA-Prompt

## Citation
```
    @inproceedings{
        huang2024ovor,
        title={{OVOR}: OnePrompt with Virtual Outlier Regularization for Rehearsal-Free Class-Incremental Learning},
        author={Huang, Wei-Cheng and Chen, Chun-Fu and Hsu, Hsiang},
        booktitle={The Twelfth International Conference on Learning Representations},
        year={2024},
        url={https://openreview.net/forum?id=FbuyDzZTPt}
    }
```