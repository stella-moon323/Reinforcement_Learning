from config import Config
from env import Environment


def main():
    config = Config()
    env = Environment()
    env.run(config)


if __name__ == "__main__":
    main()
