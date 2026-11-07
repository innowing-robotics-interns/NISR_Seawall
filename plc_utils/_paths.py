"""Make the moved analysis scripts importable from either entry point.

The scripts in this folder refer to `mini_nn`, `loop_extraction`, `rosa_main`
and to one another by their bare module names, and they resolve run folders
such as `log/...` against the repository root. Running

    python plc_utils/<script>.py
    python -m plc_utils.<script>

both put the repository root and this folder on `sys.path`, so those names and
paths keep working without editing every import.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

for _path in (_ROOT, _ROOT / 'plc_utils'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))