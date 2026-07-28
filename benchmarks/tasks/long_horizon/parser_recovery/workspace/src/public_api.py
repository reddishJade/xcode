"""稳定的公开导入入口。"""

from src.config_parser import ParseError, parse_config

__all__ = ["ParseError", "parse_config"]
