"""Launch independent science tasks sequentially on the configured GPUs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local.periodic_m2.run_tasks import main

if __name__ == '__main__':
    sys.exit(main())
