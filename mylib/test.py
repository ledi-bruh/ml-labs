import sys
import os
from pathlib import Path

if (dir := str(Path(os.getcwd()))) not in sys.path:
    sys.path.append(dir)

