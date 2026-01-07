import os
import json
import math

import torch
import torchnet as tnt

from protonets.utils import filter_opt
import protonets.utils.data as data_utils
import protonets.utils.model as model_utils


def _is_state_checkpoint(obj) -> bool:
    return isinstance(obj, dict) and "model_state" in obj


def main(opt):
    model_path = opt["model.model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model/checkpoint not found:\n  {model_path}\n"
            f"Tip: list available checkpoints with:\n"
            f"  Get-ChildItem -Path .\\results -Recurse -Filter *.pt"
        )

    # Locate opt.json (prefers alongside the model/checkpoint)
    model_opt_file = os.path.join(os.path.dirname(model_path), "opt.json")
    if not os.path.exists(model_opt_file):
        # fallback: results/opt.json
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        alt = os.path.join(repo_root, "results", "opt.json")
        if os.path.exists(alt):
            model_opt_file = alt
        else:
            raise FileNotFoundError(
                f"Could not find opt.json next to the model:\n  {model_opt_file}\n"
                f"Also checked:\n  {alt}"
            )

    with open(model_opt_file, "r") as f:
        model_opt = json.load(f)

    # Postprocess options
    model_opt["model.x_dim"] = list(map(int, model_opt["model.x_dim"].split(",")))
    model_opt["log.fields"] = model_opt["log.fields"].split(",")

    # Load either full model object or checkpoint dict
    loaded = torch.load(model_path, weights_only=False, map_location="cpu")

    if _is_state_checkpoint(loaded):
        # Rebuild model from opt.json and load weights
        model = model_utils.load(model_opt)
        model.load_state_dict(loaded["model_state"])
    else:
        model = loaded

    model.eval()

    # Construct data opts
    data_opt = {"data." + k: v for k, v in filter_opt(model_opt, "data").items()}

    # Override episodic params for evaluation if provided
    episode_fields = {
        "data.test_way": "data.way",
        "data.test_shot": "data.shot",
        "data.test_query": "data.query",
        "data.test_episodes": "data.train_episodes",
    }

    for k, v in episode_fields.items():
        if opt.get(k, 0) != 0:
            data_opt[k] = opt[k]
        elif model_opt.get(k, 0) != 0:
            data_opt[k] = model_opt[k]
        else:
            data_opt[k] = model_opt[v]

    print(
        "Evaluating {:d}-way, {:d}-shot with {:d} query examples/class over {:d} episodes".format(
            data_opt["data.test_way"],
            data_opt["data.test_shot"],
            data_opt["data.test_query"],
            data_opt["data.test_episodes"],
        )
    )

    torch.manual_seed(1234)
    if data_opt.get("data.cuda", False):
        torch.cuda.manual_seed(1234)

    data = data_utils.load(data_opt, ["test"])

    if data_opt.get("data.cuda", False):
        model.cuda()

    meters = {field: tnt.meter.AverageValueMeter() for field in model_opt["log.fields"]}
    model_utils.evaluate(model, data["test"], meters, desc="test")

    for field, meter in meters.items():
        mean, std = meter.value()
        print(
            "test {:s}: {:0.6f} +/- {:0.6f}".format(
                field, mean, 1.96 * std / math.sqrt(data_opt["data.test_episodes"])
            )
        )
