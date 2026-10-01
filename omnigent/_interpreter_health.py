"""Detect standard-library import failures and explain them without an Omnigent crash report.

When the interpreter's own standard library cannot be imported, nothing Omnigent ships
can repair it, yet Python prints a dependency traceback and the crash handler offers to
file a bug. These helpers recognise that failure and name the real cause instead.
"""

from __future__ import annotations

import linecache
import os
import sys
import sysconfig
import traceback
from types import TracebackType
from typing import NamedTuple, TextIO


class BrokenStdlibModule(NamedTuple):
    """The standard-library module whose body raised, and the line that did."""

    name: str
    filename: str
    lineno: int


def broken_stdlib_module(
    exc: BaseException, tb: TracebackType | None = None
) -> BrokenStdlibModule | None:
    """Return the stdlib module that failed to import, or ``None`` for an ordinary crash.

    Only the innermost module body on the traceback (``tb``, else ``exc.__traceback__``)
    decides: a stdlib module raising while it is imported means a damaged interpreter;
    Omnigent or a dependency importing a module this platform lacks is still a bug there.
    """
    innermost: BrokenStdlibModule | None = None
    tb = tb if tb is not None else exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        if frame.f_code.co_name == "<module>":
            name = frame.f_globals.get("__name__")
            innermost = None
            # The entry script's own body runs as ``__main__``; it is never stdlib.
            if (
                isinstance(name, str)
                and name != "__main__"
                and name.partition(".")[0] in sys.stdlib_module_names
            ):
                innermost = BrokenStdlibModule(name, frame.f_code.co_filename, tb.tb_lineno)
        tb = tb.tb_next
    return innermost


def _inside_stdlib(filename: str) -> bool:
    """Whether ``filename`` is the interpreter's own copy rather than a shadowing one."""
    if not filename or filename.startswith("<"):
        return True
    real = os.path.realpath(filename)
    paths = sysconfig.get_paths()
    return any(
        real.startswith(os.path.realpath(paths[key]) + os.sep)
        for key in ("stdlib", "platstdlib")
        if paths.get(key)
    )


def render_broken_stdlib_notice(
    exc: BaseException, module: BrokenStdlibModule, stream: TextIO | None = None
) -> None:
    """Explain that the interpreter, not Omnigent, is broken, and how to recover."""
    out = stream if stream is not None else sys.stderr
    version = ".".join(str(part) for part in sys.version_info[:3])
    lines = [
        "",
        "Omnigent cannot run: this Python installation's standard library cannot be imported.",
        "",
        f"  Python {version} ({sys.base_prefix})",
        f"  failed while importing its own module {module.name}:",
        f'    File "{module.filename}", line {module.lineno}',
    ]
    source = linecache.getline(module.filename, module.lineno).strip()
    if source:
        lines.append(f"      {source}")
    lines.extend(
        f"    {line.rstrip()}"
        for line in traceback.format_exception_only(type(exc), exc)
        if line.strip()
    )
    lines.append("")
    if _inside_stdlib(module.filename):
        lines += [
            "This is a problem with the Python installation, not with Omnigent, so there is",
            "nothing to report as an Omnigent bug. Repair or reinstall Python, then run",
            "omnigent again.",
        ]
    else:
        stdlib_dir = sysconfig.get_paths()["stdlib"]
        lines += [
            f"That file is outside this Python installation's standard library ({stdlib_dir}),",
            "so something on PYTHONPATH or in the working directory shadows the real module.",
            "There is nothing to report as an Omnigent bug; remove or fix that copy, then run",
            "omnigent again.",
        ]
    lines.append("")
    out.write("\n".join(lines) + "\n")
    out.flush()


def exit_if_stdlib_broken(exc: BaseException) -> None:
    """Exit 1 with the notice when the interpreter is at fault; otherwise return."""
    module = broken_stdlib_module(exc)
    if module is None:
        return
    render_broken_stdlib_notice(exc, module)
    raise SystemExit(1)
