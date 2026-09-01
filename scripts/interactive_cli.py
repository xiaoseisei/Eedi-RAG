"""
================================================================================
脚本名称: scripts/interactive_cli.py
业务定位: Eedi-RAG 交互式名师教研备课控制台启动入口
运行方式:
  python scripts/interactive_cli.py
================================================================================
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.cli import main

if __name__ == "__main__":
    main()
