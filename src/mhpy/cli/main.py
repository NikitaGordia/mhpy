import hydra
from omegaconf import DictConfig

from mhpy.cli.commands.initialize import init
from mhpy.utils.common import configure_logger


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logger(save_logs=False)

    if cfg.command.name == "init":
        init(cfg.command)


if __name__ == "__main__":
    main()
