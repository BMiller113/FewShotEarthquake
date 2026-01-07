import os
import json
import subprocess
import sys

from protonets.utils import format_opts, merge_dict
from protonets.utils.log import load_trace


def main(opt):
    # Resolve repo root robustly
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    # Canonical results dir
    result_dir = os.path.join(repo_root, "results")

    # Load trace to determine best epoch
    trace_file = os.path.join(result_dir, "trace.txt")
    trace_vals = load_trace(trace_file)
    best_epoch = trace_vals["val"]["loss"].argmin()

    # Load opts from training run
    model_opt_file = os.path.join(result_dir, "opt.json")
    with open(model_opt_file, "r") as f:
        model_opt = json.load(f)

    # Override previous training opts for trainval mode
    model_opt = merge_dict(
        model_opt,
        {
            # Force trainval outputs under <repo_root>/results/trainval
            "log.exp_dir": os.path.join(result_dir, "trainval"),
            "data.trainval": True,
            "train.epochs": int(best_epoch) + int(model_opt.get("train.patience", 0)),
        },
    )

    # Absolute path to run_train.py
    run_train_py = os.path.join(repo_root, "scripts", "train", "few_shot", "run_train.py")
    py = sys.executable

    subprocess.check_call([py, run_train_py] + format_opts(model_opt))
