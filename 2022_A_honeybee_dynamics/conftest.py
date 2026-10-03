"""Pytest / unittest 路径护栏：保证题包根目录可导入 ``src``。"""

from __future__ import annotations

import sys
from pathlib import Path

_PACK_ROOT = Path(__file__).resolve().parent
_root = str(_PACK_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)
