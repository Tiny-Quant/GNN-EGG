"""Run the Comparison_v3 notebook workflow outside of Jupyter.

This script mirrors the notebook cells while adding:

* Logging of all stdout/stderr output to disk
* Plot capture in a headless environment
* Basic memory-usage reporting to catch potential OOM conditions
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

# Non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

from new_src.agg_instance import (
    aggregate_instance_explanations,
    build_explainer,
    plot_eval,
    run_eval_summary,
)
from new_src.criteria import (
    BudgetPenalty,
    ClassScoreCriterion,
    EmbeddingCriterion,
    KLDivergencePenalty,
    NormPenalty,
    WeightedCriterion,
)
from new_src.dataAdapter import load_dataset
from new_src.eval import eval_summary
from new_src.explainee import fit_explainee
from new_src.ged_dataset import create_and_save_ged_dataset
from new_src.graph_level_dist import neural_approx_ged_dist
from new_src.graph_sampler import GraphSampler
from new_src.simgnn import SimGNN, train_simgnn
from new_src.trainer import Trainer
from new_src.utils import eval_plot

SEED = 123123


def set_reproducibility(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)


class LoggingStream(io.TextIOBase):
    """Redirect ``print`` output into the logging system."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        super().__init__()
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, data: str) -> int:  # type: ignore[override]
        if not isinstance(data, str):
            data = str(data)
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self.logger.log(self.level, line)
        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        if self._buffer:
            self.logger.log(self.level, self._buffer.rstrip())
            self._buffer = ""


@contextmanager
def redirect_output(logger: logging.Logger):
    """Redirect stdout/stderr so notebook-style prints reach the log file."""

    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = LoggingStream(logger, logging.INFO)
    sys.stderr = LoggingStream(logger, logging.ERROR)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = stdout, stderr


@dataclass
class MemoryUsage:
    rss_bytes: int
    total_bytes: Optional[int]

    @property
    def percent(self) -> Optional[float]:
        if self.total_bytes:
            return (self.rss_bytes / self.total_bytes) * 100
        return None

    @property
    def human_rss(self) -> str:
        return f"{self.rss_bytes / (1024 ** 3):.2f} GB"


def read_memory_usage() -> MemoryUsage:
    """Return best-effort process memory information."""

    try:
        import psutil

        proc = psutil.Process()
        rss_bytes = proc.memory_info().rss
        total_bytes = psutil.virtual_memory().total
        return MemoryUsage(rss_bytes=rss_bytes, total_bytes=total_bytes)
    except Exception:
        try:
            import resource

            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_bytes = int(rss_kb * 1024)
        except Exception:
            rss_bytes = 0

        total_bytes = None
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                    total_bytes = total_kb * 1024
                    break

        return MemoryUsage(rss_bytes=rss_bytes, total_bytes=total_bytes)


def log_memory_usage(logger: logging.Logger, context: str) -> None:
    usage = read_memory_usage()
    pct = usage.percent
    if pct is not None:
        logger.info("Memory usage %s: %s (%.2f%% of system memory)", context, usage.human_rss, pct)
        if pct > 85:
            logger.warning(
                "Memory usage above 85%% may destabilize the run; consider reducing batch sizes."
            )
    else:
        logger.info("Memory usage %s: %s", context, usage.human_rss)


class PlotSaver:
    """Replace ``plt.show`` to write figures to disk instead of displaying them."""

    def __init__(self, base_dir: Path, logger: logging.Logger) -> None:
        self.base_dir = base_dir
        self.logger = logger
        self._original_show = plt.show
        self._patched_show = None
        self.prefix = "plot"
        self.counter = 0

    def set_prefix(self, prefix: str) -> None:
        self.prefix = prefix

    def _save_and_close(self, *args, **kwargs) -> None:
        figures = [plt.figure(num) for num in plt.get_fignums()]
        if not figures:
            self.logger.info("plt.show() called but no figures are open.")
            return

        for fig in figures:
            self.counter += 1
            name = f"{self.prefix}_fig_{self.counter:03d}.png"
            path = self.base_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
            self.logger.info("Saved plot to %s", path)
            plt.close(fig)

    def __enter__(self):
        def _patched_show(*args, **kwargs):
            return self._save_and_close(*args, **kwargs)

        self._patched_show = _patched_show
        plt.show = self._patched_show  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb):
        plt.show = self._original_show  # type: ignore[assignment]
        self._patched_show = None


# ---------------------------
# Original notebook functions
# ---------------------------


def run_GNNInt(cls_idx, max_nodes, data, mean_embeds, explainee):
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
                    weight=1,
                ),
                dict(
                    key="embeds",
                    criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_idx]),
                    weight=50,
                ),
                dict(key="omega", criterion=NormPenalty(order=1), weight=1),
                dict(key="omega", criterion=NormPenalty(order=2), weight=1),
                dict(
                    key="theta_pairs",
                    criterion=KLDivergencePenalty(binary=True),
                    weight=4,
                ),
            ]
        ),
        optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
        scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
        dataset=data,
        budget_penalty=BudgetPenalty(budget=20, order=2, beta=1),
        device="cpu",
    )

    trainer.train(
        iterations=200,
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
        example.edge_attr = F.one_hot(
            example.edge_label, num_classes=len(data.EDGE_CLS)
        ).float()

    example.y = torch.tensor(cls_idx).float()
    return example


def eval_GNNInt(data, mean_embeds, explainee, ged_model):
    graphs_0 = [
        run_GNNInt(
            cls_idx=0,
            max_nodes=20,
            data=data,
            mean_embeds=mean_embeds,
            explainee=explainee,
        )
        for _ in range(5)
    ]

    graphs_1 = [
        run_GNNInt(
            cls_idx=1,
            max_nodes=20,
            data=data,
            mean_embeds=mean_embeds,
            explainee=explainee,
        )
        for _ in range(5)
    ]

    cls_split = data.split_by_class()

    dist_to_0 = neural_approx_ged_dist(cls_split[0], ged_model)
    dist_to_1 = neural_approx_ged_dist(cls_split[1], ged_model)

    eval_summary(
        explainee=explainee,
        gen_graphs_0=graphs_0,
        gen_graphs_1=graphs_1,
        obs_graphs_0=cls_split[0],
        obs_graphs_1=cls_split[1],
        dist_to_0=dist_to_0,
        dist_to_1=dist_to_1,
    )

    eval_plot(
        explainee,
        graphs_0,
        graphs_1,
        obs_graphs_0=cls_split[0],
        obs_graphs_1=cls_split[1],
        ged_model=ged_model,
        dataset=data,
        layout="kamada",
    )


def eval_agg_explainer(explainee, data, ged_model):
    explainer = build_explainer(
        explainee,
        algorithm="gnnexplainer",  # try "pgexplainer", Captum variants, etc.
        algorithm_kwargs={"epochs": 50},
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
    )

    aggregation = aggregate_instance_explanations(
        data,
        explainer,
        explainee=explainee,
        strategy="wl_topk",
        strategy_kwargs={"wl_hops": 2, "top_p": 0.25, "min_edges": 4},
    )

    cls_split = data.split_by_class()

    dist_to_0 = neural_approx_ged_dist(cls_split[0], ged_model)
    dist_to_1 = neural_approx_ged_dist(cls_split[1], ged_model)

    run_eval_summary(
        explainee,
        aggregation,
        observed_class_0=cls_split[0],
        observed_class_1=cls_split[1],
        dist_to_0=dist_to_0,
        dist_to_1=dist_to_1,
    )

    plot_eval(
        explainee,
        aggregation,
        observed_class_0=cls_split[0],
        observed_class_1=cls_split[1],
        ged_model=ged_model,
        dataset=data,
        max_pairs=3,
    )


def prep_data(dataset_name):
    data = load_dataset(dataset_name)
    cls_split = data.split_by_class()

    explainee, _ = fit_explainee(
        dataset_name=dataset_name,
        root="data",
        hidden=64,
        layers=3,
        dropout=0.2,
        epochs=100,
        batch_size=16,
        lr=1e-3,
        device="cpu",
    )

    mean_embeds = [
        torch.cat(
            [explainee(batch.to("cpu"))["embeds"] for batch in DataLoader(subset, batch_size=64)]
        )
        .mean(dim=0)
        for subset in cls_split
    ]
    mean_embeds = [m.detach() for m in mean_embeds]

    ged_dataset = create_and_save_ged_dataset(
        data,
        out_path=None,
        n_perturbations=1,
        node_add_prob=0.5,
        node_remove_prob=0.5,
        edge_add_prob=0.1,
        edge_remove_prob=0.1,
    )

    print(f"Created GED dataset with {len(ged_dataset)} pairs.")

    model = SimGNN(in_dim=len(data.NODE_CLS))

    train_simgnn(
        model=(m := model),
        data=ged_dataset,
        batch_size=16,
        optimizer=(o := torch.optim.Adam(m.parameters(), lr=1e-3)),
        scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
        epochs=5,
        device="cpu",
    )

    return data, explainee, mean_embeds, model


# ---------------------------
# Runner utilities
# ---------------------------

def configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("comparison_v3")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if the script is re-imported
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.__stdout__)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def run_workflow(datasets: Iterable[str], output_dir: Path) -> None:
    set_reproducibility()

    log_path = output_dir / "comparison_v3.log"
    plots_dir = output_dir / "plots"

    logger = configure_logger(log_path)
    logger.info("Starting Comparison_v3 script. Logging to %s", log_path)

    with redirect_output(logger), PlotSaver(plots_dir, logger) as plot_saver:
        for dataset in datasets:
            logger.info("\n===== Running dataset: %s =====", dataset)
            plot_saver.set_prefix(dataset.lower())
            dataset_dir = plots_dir / dataset.lower()
            dataset_dir.mkdir(parents=True, exist_ok=True)
            plot_saver.base_dir = dataset_dir

            log_memory_usage(logger, f"before preparing {dataset}")
            try:
                data, explainee, mean_embeds, model = prep_data(dataset)
                log_memory_usage(logger, f"after preparing {dataset}")

                logger.info("Running GNNInterpreter for %s", dataset)
                eval_GNNInt(
                    data=data,
                    mean_embeds=mean_embeds,
                    explainee=explainee,
                    ged_model=model,
                )
                log_memory_usage(logger, f"after GNNInterpreter for {dataset}")

                logger.info("Running aggregated explainer for %s", dataset)
                plot_saver.set_prefix(f"{dataset.lower()}_agg")
                eval_agg_explainer(
                    data=data,
                    explainee=explainee,
                    ged_model=model,
                )
                log_memory_usage(logger, f"after aggregated explainer for {dataset}")
            except Exception:
                logger.exception("Dataset %s failed due to an unexpected error.", dataset)

    logger.info("Completed all datasets.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Comparison_v3 workflow without Jupyter.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["MUTAG", "BA-2MOTIFS", "PROTEINS"],
        help="Datasets to evaluate (defaults to the three used in the notebook).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revisions/results/comparison_v3"),
        help="Directory where logs and plots will be written.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_workflow(args.datasets, args.output_dir)
