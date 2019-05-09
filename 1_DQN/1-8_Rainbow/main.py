import logzero
import os

from config import Config
from agent import Agent


def setup_logger(config: Config):
    log_dir = config.log.log_dir
    if log_dir is not None and not os.path.exists(log_dir):
        os.mkdir(config.log.log_dir)

    return logzero.setup_logger(
        config.log.name,
        config.log.logfile,
        config.log.level,
        config.log.formatter,
        config.log.maxBytes,
        config.log.backupCount,
        config.log.fileLoglevel,
        config.log.disableStderrLogger
    )


def main():
    config = Config()
    logger = setup_logger(config)
    logger.info("Start DQN")
    agent = Agent(config, logger)
    agent.start()


if __name__ == "__main__":
    main()
