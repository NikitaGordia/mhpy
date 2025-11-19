import sys

import hydra
from omegaconf import DictConfig

from mhpy.cli.commands.initialize import init
from mhpy.utils.common import configure_logger


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logger(save_logs=False)

    try:
        init(cfg)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
