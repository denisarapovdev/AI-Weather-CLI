import asyncio
import sys

from .cli import run_cli
from .config import CLI_FATAL_ERROR, validate_config


def main() -> None:
    """Entry point that validates config and runs the CLI."""
    validate_config()

    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(CLI_FATAL_ERROR.format(error=e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
