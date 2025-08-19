from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import numpy as np
import pandas as pd
import typer
from uncalibrated_reasoning.utils import get_cache_dir
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os



def run_data_parallel(
    pretrained_model_name: str | None = None,
    experiment_name: str | None = None,
    step: int | None = None,
    data_key: str = "replogle_k562_essential_cnmf",
    prompt_key: str = "hit_probability_v2",
    split_dir: str = "split",
    split: str = "val",
    n_samples: int = 4,
    debug: bool = False,
):
    n_gpus = torch.cuda.device_count()
    futures = []
    with ProcessPoolExecutor(max_workers=n_gpus) as pool:
        for i in range(n_gpus):
            future = pool.submit(
                run_helper,
                pretrained_model_name,
                experiment_name,
                step,
                data_key,
                prompt_key,
                split_dir,
                split,
                n_samples,
                debug,
                i,
                n_gpus,
            )
            futures.append(future)
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            print(future.result())

def run_helper(
    pretrained_model_name: str | None = None,
    experiment_name: str | None = None,
    step: int | None = None,
    data_key: str = "replogle_k562_essential_cnmf",
    prompt_key: str = "hit_probability_v2",
    split_dir: str = "split",
    split: str = "val",
    n_samples: int = 4,
    debug: bool = False,
    job_idx: int = 0,
    total_jobs: int = 1,
):
    torch.set_float32_matmul_precision("high")
    print(job_idx, total_jobs)

    path = get_cache_dir() / "task_data" / data_key / split_dir / f"{split}.pq"
    df = pd.read_parquet(path)

    if total_jobs > 1:
        job_size = int(np.ceil(df.shape[0] / total_jobs))
        df = df.iloc[job_idx * job_size: (job_idx + 1) * job_size]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(job_idx)

    with open(f"prompts/{prompt_key}.prompt") as f:
        prompt_template = f.read()

    if experiment_name is not None:
        model_cache_dir = get_cache_dir() / "experiments" / experiment_name / f"global_step_{step}" / "actor_merged"
    else:
        model_cache_dir = get_cache_dir() / "models" / pretrained_model_name

    tokenizer = AutoTokenizer.from_pretrained(model_cache_dir)

    prompts = []
    for i in range(df.shape[0]):
        prompt = prompt_template.format(**df.iloc[i].to_dict())
        messages = [
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(text)

    df["prompt"] = prompts

    if debug:
        df = df.head(10)

    df = pd.concat([df] * n_samples)

    prompts = list(df["prompt"])

    model = LLM(str(model_cache_dir), enforce_eager=True)

    sampling_params = SamplingParams(
        n=1,
        temperature=1, 
        top_p=1., 
        min_tokens=0,
        max_tokens=10_000,
        stop=["</answer>"], 
        include_stop_str_in_output=True,
    )

    outputs = model.generate(prompts, sampling_params)

    results = defaultdict(list)
    for i in range(len(outputs)):
        text = outputs[i].outputs[0].text
        results["thinking_content"].append(extract_between(text, "<think>", "</think>"))
        results["answer"].append(extract_between(text, "<answer>", "</answer>"))
    results = pd.DataFrame(results)


    df = df.reset_index(drop=True)
    df["thinking_content"] = results["thinking_content"]
    df["answer"] = results["answer"]

    if pretrained_model_name is None:
        outdir = model_cache_dir.parent / data_key
    else:
        outdir = get_cache_dir() / "baselines" / data_key    
    
    subdir = [prompt_key, split_dir, split]
    if debug:
        subdir.append("debug")
    subdir = "-".join(subdir)

    outdir = outdir / subdir
    outdir.mkdir(exist_ok=True, parents=True)

    filename = f"preds-{job_idx + 1}-{total_jobs}.pq"
    path = outdir / filename
    print(path)
    df.to_parquet(path)

def extract_between(text, start, end):
    pieces = text.split(start)
    if len(pieces) < 2:
        return None
    else:
        return pieces[1].split(end)[0]

if __name__ == "__main__":
    typer.run(run_data_parallel)
