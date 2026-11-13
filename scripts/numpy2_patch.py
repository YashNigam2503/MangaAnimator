"""Patches a clone of comic-text-detector so it runs under NumPy 2.

NumPy 2 removed a set of old aliases (np.bool8, np.float_, np.int0, ...) that the
detector still uses. The tools/ folder is not part of this repository, so the fix
is applied to the clone after it is fetched:

    python scripts/numpy2_patch.py tools/comic-text-detector

Running it again changes nothing. .ipynb_checkpoints folders are deleted, and a
note is added at the top of the folder's README.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

# removed alias -> the name that exists in both NumPy 1 and NumPy 2
ALIASES = {
    "bool8": "bool_",
    "float_": "float64",
    "int0": "intp",
    "uint0": "uintp",
    "object0": "object_",
    "str0": "str_",
    "bytes0": "bytes_",
    "unicode_": "str_",
    "complex_": "complex128",
    "cfloat": "complex128",
    "Inf": "inf",
    "Infinity": "inf",
    "infty": "inf",
    "PINF": "inf",
    "NaN": "nan",
    "NAN": "nan",
}
# "np.float_" must not match inside "np.float_power", nor "np.Inf" inside "np.Inferior".
ALIAS_RE = re.compile(r"\b(np|numpy)\.(" + "|".join(ALIASES) + r")(?![A-Za-z0-9_])")
# A one-line tuple of NumPy types, such as NP_FLOAT_TYPES = (np.float_, np.float16, ...)
TYPE_TUPLE_RE = re.compile(r"^(\s*\w+\s*=\s*)\(((?:\s*(?:np|numpy)\.\w+\s*,?)+)\)(\s*)$")
NOTE = ("> **Patched for NumPy 2** by MangaAnimator's `scripts/numpy2_patch.py`: the aliases NumPy 2 removed "
        "(`np.bool8`, `np.float_`, `np.int0`, ...) are replaced by their current names.\n\n")
SKIP_DIRS = {".git", ".ipynb_checkpoints", "__pycache__"}


def patch_source(text: str) -> str:
    """Source code with every removed alias replaced."""
    lines = []
    for line in text.split("\n"):
        new = ALIAS_RE.sub(lambda m: f"{m.group(1)}.{ALIASES[m.group(2)]}", line)
        match = TYPE_TUPLE_RE.match(new) if new != line else None
        if match:
            # The replacement can repeat a type already in the tuple (np.float_ was np.float64): keep one.
            items = list(dict.fromkeys(i.strip() for i in match.group(2).split(",") if i.strip()))
            body = ", ".join(items) + ("," if len(items) == 1 else "")
            new = f"{match.group(1)}({body}){match.group(3)}"
        lines.append(new)
    return "\n".join(lines)


def find_aliases(folder: Path) -> list[str]:
    """Every remaining use of a removed alias, as 'file:line: code'."""
    found = []
    for path in python_files(folder):
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if ALIAS_RE.search(line):
                found.append(f"{path.relative_to(folder).as_posix()}:{number}: {line.strip()}")
    return found


def python_files(folder: Path):
    for path in sorted(folder.rglob("*.py")):
        if not SKIP_DIRS & set(path.relative_to(folder).parts):
            yield path


def patch_folder(folder: Path) -> list[str]:
    """Patch every Python file in the folder. Returns the files that changed."""
    for checkpoints in sorted(folder.rglob(".ipynb_checkpoints")):
        if checkpoints.is_dir():
            shutil.rmtree(checkpoints)
    changed = []
    for path in python_files(folder):
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="surrogateescape")
        patched = patch_source(text)
        if patched != text:
            path.write_bytes(patched.encode("utf-8", errors="surrogateescape"))
            changed.append(path.relative_to(folder).as_posix())
    readme = folder / "README.md"
    if readme.is_file():
        text = readme.read_bytes().decode("utf-8", errors="surrogateescape")
        if "Patched for NumPy 2" not in text:
            readme.write_bytes((NOTE + text).encode("utf-8", errors="surrogateescape"))
            changed.append("README.md")
    return changed


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    folder = Path(argv[0])
    if not folder.is_dir():
        print(f"ERROR: folder not found: {folder}")
        return 1
    changed = patch_folder(folder)
    left = find_aliases(folder)
    if left:
        print("ERROR: removed NumPy aliases are still present:\n  " + "\n  ".join(left))
        return 1
    print(f"NumPy 2 patch: {len(changed)} file(s) changed" + (f" ({', '.join(changed)})" if changed else ", already patched"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
