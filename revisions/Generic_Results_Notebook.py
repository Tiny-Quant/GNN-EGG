#!/usr/bin/env python3
"""Scripted version of ``Generic_Results_Notebook.ipynb``.

This script mirrors the notebook flow while capturing all stdout/stderr, plots,
Ray Tune summaries, and intermediate metrics to disk. Memory usage (CPU RSS and
CUDA allocations when available) is logged throughout to help diagnose OOM or
kernel death issues.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import itertools
import json
import logging
import os
import resource
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

# Ensure headless plotting and deterministic, repeatable images.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from torch_geometric.utils import from_networkx  # noqa: E402

# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------


class Tee(io.TextIOBase):
    """Tee writes to multiple streams (e.g., console + file)."""

    def __init__(self, *streams: Iterable[io.TextIOBase]):
        self.streams = streams

    def write(self, data: str) -> int:  # type: ignore[override]
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        for stream in self.streams:
            stream.flush()


def format_bytes(num_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:0.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:0.2f} PB"


def current_rss_bytes() -> float:
    """Return current resident set size using /proc/self/statm if available."""

    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            parts = f.read().split()
        rss_pages = int(parts[1])
        return rss_pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, IndexError, ValueError, OSError):
        return float("nan")


def log_memory(logger: logging.Logger, note: str) -> None:
    """Log CPU and CUDA memory usage to the configured logger."""

    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_bytes = max_rss_kb * 1024
    rss_now = current_rss_bytes()

    cuda_msg = ""
    if torch.cuda.is_available():
        cuda_alloc = format_bytes(torch.cuda.memory_allocated())
        cuda_reserve = format_bytes(torch.cuda.memory_reserved())
        cuda_msg = f", cuda_alloc={cuda_alloc}, cuda_reserved={cuda_reserve}"

    logger.info(
        "Memory snapshot (%s): current=%s, max=%s%s",
        note,
        format_bytes(rss_now) if not np.isnan(rss_now) else "n/a",
        format_bytes(max_rss_bytes),
        cuda_msg,
    )


# ---------------------------------------------------------------------------
# Plot capture utilities
# ---------------------------------------------------------------------------


def install_plot_saver(output_dir: Path, logger: logging.Logger) -> None:
    """Hijack ``plt.show`` so every figure is saved and closed."""

    counter = itertools.count()

    def save_and_close(fig=None, *args, **kwargs):  # type: ignore[override]
        fig = fig or plt.gcf()
        idx = next(counter)
        path = output_dir / f"figure_{idx:03d}.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved plot to %s", path)
        plt.close(fig)

    plt.show = save_and_close  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Core pipeline mirroring the notebook
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Generic Results pipeline")
    parser.add_argument("--dataset-name", default="", help="Dataset name for load_dataset")
    parser.add_argument("--device", default="cuda", help="Device for training (e.g., cpu, cuda)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for logs and artifacts; defaults to revisions/results/<timestamp>.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = output_dir / f"run_{timestamp}.log"
    stdout_path = output_dir / f"stdout_{timestamp}.log"

    original_stdout, original_stderr = sys.stdout, sys.stderr
    stdout_file = open(stdout_path, "w", encoding="utf-8", buffering=1)
    tee_stdout = Tee(original_stdout, stdout_file)
    tee_stderr = Tee(original_stderr, stdout_file)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    logger = logging.getLogger("generic_results")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_path)
    stream_handler = logging.StreamHandler(original_stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    logger.info("Logging to %s", log_path)
    logger.info("Stdout/Stderr capture at %s", stdout_path)
    return logger


def configure_seeds(seed: int = 123123) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:  # noqa: C901 - preserve notebook flow
    args = parse_args()
    output_dir = args.output_dir or repo_root() / "revisions" / "results" / "generic_results_runs"
    output_dir = output_dir / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = setup_logging(output_dir)
    install_plot_saver(output_dir, logger)

    logger.info("Starting Generic Results pipeline")
    configure_seeds()
    log_memory(logger, "initial")

    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Import after sys.path manipulation
    import importlib
    from new_src.dataAdapter import load_dataset
    from new_src.explainee import fit_explainee
    from new_src.ged_dataset import create_and_save_ged_dataset, plot_graph_pair
    from new_src.simgnn import SimGNN, train_simgnn

    DATASET_NAME = args.dataset_name
    config = {
        "DATASET_NAME": DATASET_NAME,
        "device": args.device,
        "explainee": {"epochs": 200, "batch_size": 64, "lr": 1e-3},
        "ged_model": {"n_perds": 4, "epochs": 200, "batch_size": 64},
        "tune": {
            "n_trials": 16,
            "max_nodes": 24,
            "iters_per_sample": 16,
            "samples_per_trial": 24,
        },
        "train": {"iters_per_sample": 200, "samples_per_trial": 100},
    }

    logger.info("Config: %s", json.dumps(config, indent=2))

    # ------------------------------------------------------------------
    # Data loading and explainee training
    # ------------------------------------------------------------------
    importlib.reload(importlib.import_module("new_src.dataAdapter"))
    data = load_dataset(config.get("DATASET_NAME"))
    cls_split = data.split_by_class()
    logger.info("Loaded dataset %s with %d classes", DATASET_NAME, len(cls_split))
    log_memory(logger, "after dataset load")

    importlib.reload(importlib.import_module("new_src.explainee"))
    explainee, _ = fit_explainee(
        dataset_name=config.get("DATASET_NAME"),
        root="data",
        hidden=64,
        layers=3,
        dropout=0.2,
        epochs=config["explainee"]["epochs"],
        batch_size=config["explainee"]["batch_size"],
        lr=config["explainee"]["lr"],
        device=config.get("device"),
    )
    logger.info("Finished explainee training")
    log_memory(logger, "after explainee training")

    mean_embeds = [
        torch.cat(
            [explainee(batch.to(config.get("device")))['embeds'] for batch in DataLoader(subset, batch_size=64)]
        ).mean(dim=0)
        for subset in cls_split
    ]
    mean_embeds = [m.detach() for m in mean_embeds]
    logger.info("Computed mean embeddings for %d classes", len(mean_embeds))
    log_memory(logger, "after embeddings")

    # ------------------------------------------------------------------
    # GED dataset + visualization
    # ------------------------------------------------------------------
    importlib.reload(importlib.import_module("new_src.ged_dataset"))
    ged_dataset = create_and_save_ged_dataset(
        data,
        out_path=None,
        n_perturbations=config["ged_model"]["n_perds"],
        node_add_prob=0.5,
        node_remove_prob=0.5,
        edge_add_prob=0.1,
        edge_remove_prob=0.1,
    )
    logger.info("Created GED dataset with %d pairs", len(ged_dataset))
    log_memory(logger, "after GED dataset")

    for i in range(min(5, len(ged_dataset))):
        g1, g2, ged_norm, _, _ = ged_dataset[i]
        raw_ged = getattr(g1, "raw_ged", ged_norm)
        plot_graph_pair(g1, g2, label=raw_ged)
        logger.info("Plotted GED pair %d (label=%s)", i, raw_ged)

    # ------------------------------------------------------------------
    # SimGNN training
    # ------------------------------------------------------------------
    importlib.reload(importlib.import_module("new_src.simgnn"))
    model = SimGNN(in_dim=len(data.NODE_CLS))
    train_simgnn(
        model=(m := model),
        data=ged_dataset,
        batch_size=config["ged_model"]["batch_size"],
        optimizer=(o := torch.optim.Adam(m.parameters(), lr=1e-3)),
        scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
        epochs=config["ged_model"]["epochs"],
        device=config.get("device"),
    )
    logger.info("Finished SimGNN training")
    log_memory(logger, "after SimGNN training")

    # ------------------------------------------------------------------
    # Generator training helpers
    # ------------------------------------------------------------------
    def train_generator(
        cls_idx,
        max_nodes,
        iters_per_sample,
        data,
        mean_embeds,
        explainee,
        ged_model,
        w_pred=1,
        w_embed=1,
        w_ged=1,
        w_mcs=1,
        w_spec=1,
        w_wl=1,
        use_omega=False,
    ):
        from new_src.graph_sampler import GraphSampler
        from new_src.trainer import Trainer
        from new_src.criteria import (
            WeightedCriterion,
            ClassScoreCriterion,
            EmbeddingCriterion,
            BudgetPenalty,
        )
        from new_src.graph_level_dist import (
            neural_approx_ged_dist,
            mcs_soft_graph_dist,
            spectral_dist,
            wl_graph_kernel_dist,
        )

        trainer = Trainer(
            sampler=(
                s := GraphSampler(
                    max_nodes=max_nodes,
                    num_node_cls=len(data.NODE_CLS),
                    num_edge_cls=len(data.EDGE_CLS),
                    temperature=0.15,
                    learn_node_feat=len(data.NODE_CLS) > 0,
                    learn_edge_feat=len(data.EDGE_CLS) > 0,
                )
            ),
            discriminator=explainee,
            criterion=WeightedCriterion(
                [
                    dict(
                        key="logits",
                        criterion=ClassScoreCriterion(class_idx=cls_idx, mode="maximize"),
                        weight=w_pred,
                    ),
                    dict(
                        key="embeds",
                        criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_idx]),
                        weight=w_embed,
                    ),
                    dict(
                        key="cont_data",
                        criterion=neural_approx_ged_dist(
                            data=data, model=ged_model, explainee=explainee, cls_idx=cls_idx, use_omega=use_omega
                        ),
                        weight=w_ged,
                    ),
                    dict(
                        key="cont_data",
                        criterion=mcs_soft_graph_dist(data=data, explainee=explainee, cls_idx=cls_idx, use_omega=use_omega),
                        weight=w_mcs,
                    ),
                    dict(
                        key="cont_data",
                        criterion=spectral_dist(data=data, explainee=explainee, cls_idx=cls_idx, use_omega=use_omega),
                        weight=w_spec,
                    ),
                    dict(
                        key="cont_data",
                        criterion=wl_graph_kernel_dist(
                            data=data, explainee=explainee, cls_idx=cls_idx, use_omega=use_omega
                        ),
                        weight=w_wl,
                    ),
                ]
            ),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=data,
            budget_penalty=BudgetPenalty(budget=20, order=2, beta=1),
            device=config.get("device"),
        )

        trainer.train(
            iterations=iters_per_sample,
            target_probs={cls_idx: (0.9, 1.0)},
            target_size=30,
            w_budget_init=0.5,
            w_budget_inc=1.1,
            w_budget_dec=0.95,
            k_samples=32,
        )

        example = trainer.evaluate(threshold=0.5)
        example = from_networkx(example)

        if "label" in example:
            example.x = F.one_hot(example.label, num_classes=len(data.NODE_CLS)).float()

        if "edge_label" in example:
            example.edge_attr = F.one_hot(example.edge_label, num_classes=len(data.EDGE_CLS)).float()

        example.y = torch.tensor(cls_idx).float()
        return example

    # ------------------------------------------------------------------
    # Ray Tune setup
    # ------------------------------------------------------------------
    import ray
    from ray import tune

    ray.shutdown()
    ray.init(
        runtime_env={"py_modules": [str(Path(__file__).resolve().parent / "new_src")]},
        num_gpus=1,
    )

    data_ref = ray.put(data)
    mean_embeds_ref = ray.put(mean_embeds)
    cls_split_ref = ray.put(data.split_by_class())

    def tune_generator(
        config,
        *,
        explainee,
        model,
        max_nodes,
        gen_samples,
        iters_per_sample,
    ):
        from new_src.eval import eval_summary
        from new_src.graph_level_dist import neural_approx_ged_dist

        data = ray.get(data_ref)
        mean_embeds = ray.get(mean_embeds_ref)
        cls_split = ray.get(cls_split_ref)

        w_pred = float(config["pred_weight"])
        w_embed = float(config["embed_weight"])
        w_ged = float(config["ged_weight"])
        w_mcs = float(config["mcs_weight"])
        w_spec = float(config["spectral_weight"])
        w_wl = float(config["wl_weight"])
        use_omega = bool(config["use_omega"])

        graphs_0 = [
            train_generator(
                cls_idx=0,
                max_nodes=max_nodes,
                iters_per_sample=iters_per_sample,
                data=data,
                explainee=explainee,
                mean_embeds=mean_embeds,
                ged_model=model,
                w_pred=w_pred,
                w_embed=w_embed,
                w_ged=w_ged,
                w_mcs=w_mcs,
                w_spec=w_spec,
                w_wl=w_wl,
                use_omega=use_omega,
            )
            for _ in range(gen_samples)
        ]

        graphs_1 = [
            train_generator(
                cls_idx=1,
                max_nodes=max_nodes,
                iters_per_sample=iters_per_sample,
                data=data,
                explainee=explainee,
                mean_embeds=mean_embeds,
                ged_model=model,
                w_pred=w_pred,
                w_embed=w_embed,
                w_ged=w_ged,
                w_mcs=w_mcs,
                w_spec=w_spec,
                w_wl=w_wl,
                use_omega=use_omega,
            )
            for _ in range(gen_samples)
        ]

        dist_to_0 = neural_approx_ged_dist(cls_split[0], model)
        dist_to_1 = neural_approx_ged_dist(cls_split[1], model)

        score = eval_summary(explainee, graphs_0, graphs_1, cls_split[0], cls_split[1], dist_to_0, dist_to_1)
        tune.report(score=float(score))
        return {"score": float(score)}

    possible_weight = [0.0, 1e-2, 1e-1, 1.0, 1e1, 1e2]
    search_space = {
        "pred_weight": tune.choice(possible_weight),
        "embed_weight": tune.choice(possible_weight),
        "ged_weight": tune.choice(possible_weight),
        "mcs_weight": tune.choice(possible_weight),
        "spectral_weight": tune.choice(possible_weight),
        "wl_weight": tune.choice(possible_weight),
        "use_omega": tune.choice([True, False]),
    }

    trainable = tune.with_parameters(
        tune_generator,
        explainee=explainee,
        model=model,
        max_nodes=config["tune"]["max_nodes"],
        gen_samples=config["tune"]["samples_per_trial"],
        iters_per_sample=config["tune"]["iters_per_sample"],
    )

    gen_tuner = tune.Tuner(
        trainable=tune.with_resources(trainable, resources={"gpu": 1}),
        param_space=search_space,
        tune_config=tune.TuneConfig(num_samples=config["tune"]["n_trials"], metric="score", mode="max"),
    )

    tune_results = gen_tuner.fit()
    logger.info("Finished Ray Tune search")
    log_memory(logger, "after Ray Tune")

    best_config = tune_results.get_best_result().config
    logger.info("Best configuration: %s", best_config)

    df = tune_results.get_dataframe()
    df_path = output_dir / "ray_tune_results.csv"
    df.to_csv(df_path, index=False)
    logger.info("Saved Ray Tune dataframe to %s", df_path)
    logger.info("Ray Tune results head:\n%s", df.head().to_string())

    df_sorted = df.reset_index(drop=True)
    plt.figure(figsize=(8, 4))
    plt.plot(df_sorted.index.to_numpy(), df_sorted["score"].to_numpy(), marker="o")
    plt.xlabel("Trial")
    plt.ylabel("score")
    plt.title("Ray Tune score across trials")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Final generator training + evaluation
    # ------------------------------------------------------------------
    graphs_0 = [
        train_generator(
            cls_idx=0,
            max_nodes=config["tune"]["max_nodes"],
            iters_per_sample=config["train"]["iters_per_sample"],
            data=data,
            mean_embeds=mean_embeds,
            explainee=explainee,
            ged_model=model,
            w_pred=float(best_config["pred_weight"]),
            w_embed=float(best_config["embed_weight"]),
            w_ged=float(best_config["ged_weight"]),
            w_mcs=float(best_config["mcs_weight"]),
            w_spec=float(best_config["spectral_weight"]),
            w_wl=float(best_config["wl_weight"]),
            use_omega=bool(best_config["use_omega"]),
        )
        for _ in range(config["train"]["samples_per_trial"])
    ]

    graphs_1 = [
        train_generator(
            cls_idx=1,
            max_nodes=config["tune"]["max_nodes"],
            iters_per_sample=config["train"]["iters_per_sample"],
            data=data,
            mean_embeds=mean_embeds,
            explainee=explainee,
            ged_model=model,
            w_pred=float(best_config["pred_weight"]),
            w_embed=float(best_config["embed_weight"]),
            w_ged=float(best_config["ged_weight"]),
            w_mcs=float(best_config["mcs_weight"]),
            w_spec=float(best_config["spectral_weight"]),
            w_wl=float(best_config["wl_weight"]),
            use_omega=bool(best_config["use_omega"]),
        )
        for _ in range(config["train"]["samples_per_trial"])
    ]

    log_memory(logger, "after generator training")

    import new_src.eval as eval_module
    importlib.reload(eval_module)
    from new_src.eval import eval_summary

    summary_score = eval_summary(
        explainee=explainee,
        gen_graphs_0=graphs_0,
        gen_graphs_1=graphs_1,
        obs_graphs_0=cls_split[0],
        obs_graphs_1=cls_split[1],
    )
    logger.info("Eval summary score: %s", summary_score)

    import new_src.utils as utils_module
    importlib.reload(utils_module)
    from new_src.utils import eval_plot

    eval_plot(
        explainee,
        graphs_0,
        graphs_1,
        obs_graphs_0=cls_split[0],
        obs_graphs_1=cls_split[1],
        ged_model=model,
        dataset=data,
        layout="kamada",
    )
    log_memory(logger, "after evaluation")

    logger.info("Artifacts saved to %s", output_dir)


if __name__ == "__main__":
    main()
