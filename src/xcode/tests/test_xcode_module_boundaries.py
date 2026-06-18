"""模块边界回归测试。"""

from __future__ import annotations

from pathlib import Path
import unittest


class XcodeModuleBoundaryTests(unittest.TestCase):
    """验证已删除的模块边界不会被重新引入。"""

    def test_experimental_package_is_absent(self) -> None:
        """禁止恢复 experimental package 或对应 import。"""
        package_root = Path(__file__).resolve().parents[1]
        experimental_name = "experimental"
        self.assertFalse((package_root / experimental_name).exists())

        import_name = "xcode." + experimental_name
        imported_by: list[Path] = []
        for source_path in package_root.rglob("*.py"):
            if source_path == Path(__file__).resolve():
                continue
            source = source_path.read_text(encoding="utf-8")
            if import_name in source:
                imported_by.append(source_path.relative_to(package_root))

        self.assertEqual(imported_by, [])


if __name__ == "__main__":
    unittest.main()
