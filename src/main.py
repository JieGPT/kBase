from src.core.config import ConfigManager
from src.cli.commands import CLI


def main():
    config = ConfigManager()
    cli = CLI(config)
    cli.run()


if __name__ == "__main__":
    main()
