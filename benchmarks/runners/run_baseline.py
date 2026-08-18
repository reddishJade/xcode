"""运行保留完整历史、关闭压缩与 surface replacement 的基线组。"""

from benchmarks.runners._cli import run_variant_main


if __name__ == "__main__":
    run_variant_main("baseline")
