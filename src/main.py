import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import ConfigManager
from src.cli.commands import CLI


def main():
    config = ConfigManager()
    cli = CLI(config)
    cli.run()


if __name__ == "__main__":
    main()
