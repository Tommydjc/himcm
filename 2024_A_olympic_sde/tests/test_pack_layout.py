"""题包目录契约：复制模板后这些路径必须存在。"""

from __future__ import annotations

import unittest
from pathlib import Path

PACK_ROOT: Path = Path(__file__).resolve().parents[1]
REQUIRED_DIRS: tuple[str, ...] = (
    "data/raw",
    "data/processed",
    "src",
    "experiments",
    "results",
    "logs",
    "tests",
    "paper",
    "paper_figures",
    "prompts",
    "legacy",
)


class TestPackLayout(unittest.TestCase):
    """检查单题空壳目录是否齐全。"""

    def test_required_directories_exist(self) -> None:
        """每个契约目录都是存在的文件夹。"""
        for rel in REQUIRED_DIRS:
            path = PACK_ROOT / rel
            self.assertTrue(path.is_dir(), msg=f"missing directory: {rel}")


if __name__ == "__main__":
    unittest.main()
