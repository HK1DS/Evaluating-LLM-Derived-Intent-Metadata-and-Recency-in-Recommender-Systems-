"""Start the test deadline after importing the stage on slower test machines."""
import importlib
import os
from pathlib import Path
import sys
import time

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    stage_name, seconds = sys.argv[1:]
    stage = importlib.import_module(stage_name)
    sys.argv = [stage.__file__]
    os.environ["IKGR_LLM_DEADLINE"] = str(time.time() + float(seconds))
    stage.main()
