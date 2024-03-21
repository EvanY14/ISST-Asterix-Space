import os
from pathlib import Path

import sys
ISST_DIR = str(Path(os.getcwd()).parent.parent)
sys.path.append(ISST_DIR)

import ISST