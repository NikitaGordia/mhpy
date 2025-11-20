from pathlib import Path
import sys
import time

import git
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.multiprocessing import Queue
import wandb


class GitStatusError(Exception):
    """Raised when the git repository is not in a clean state."""

    pass


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


def get_wandb_run(run_path: str) -> wandb.Run:
    return wandb.Api().run(run_path)


def get_forked_run(run: wandb.Run, id: int) -> wandb.Run:
    path = Path(run.path).parent / str(id)
    return get_wandb_run(str(path))


def track_permormance(run: wandb.Run, it: int, total_runtime: float, step_timer: Timer, cpu_timer: Timer, queue: Queue):
    run.log(
        {
            "performance/step_runtime": step_timer.interval,
            "performance/step_runtime_cum": total_runtime,
            "performance/data_latency": cpu_timer.interval,
            "performance/data_queue_size": queue.qsize(),
        },
        step=it,
    )


def capture_args(cfg: dict) -> dict:
    cfg["command"] = " ".join(sys.argv)
    return cfg


def config_from_run(run: wandb.Run) -> DictConfig:
    return OmegaConf.create(dict(run.config))


def assert_clean_git(repo_path=".", project_name="my_project"):
    config_dir = Path(f"src/{project_name}/config")

    try:
        repo = git.Repo(repo_path)
    except git.InvalidGitRepositoryError:
        logger.error("This directory is not a Git repository. Skipping check.")
        return

    for diff_item in repo.index.diff(None):
        file_path = Path(diff_item.a_path)
        if not file_path.is_relative_to(config_dir):
            raise GitStatusError()

    for untracked_path_str in repo.untracked_files:
        file_path = Path(untracked_path_str)
        if not file_path.is_relative_to(config_dir):
            raise GitStatusError()
