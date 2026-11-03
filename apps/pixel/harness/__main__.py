import sys

from .cli import main

if __name__ == "__main__":  # multiprocessing spawn re-imports this module; never re-run the CLI there
    sys.exit(main())        # the verdict is the exit code: 0 PASS, 1 FAIL, 2 INCOMPLETE
