"""运行启用分层压缩、durable surface 与恢复的 Xcode 组。"""

from benchmarks.runners._cli import run_variant_main


if __name__ == "__main__":
    run_variant_main("xcode")
