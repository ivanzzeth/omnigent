"""Tell a broken Python installation apart from an Omnigent crash.

A standard-library module that fails to import (one user's ``asyncio/runners.py`` read
``import aiohttp``) is nothing Omnigent can repair, yet Python shows a dependency
traceback and the crash handler offers to file a bug. Name the real cause instead.
"""

from __future__ import annotations

import linecache
import sys
import traceback
from typing import NamedTuple, TextIO


class BrokenStdlibModule(NamedTuple):
    """The standard-library module whose body raised, and the line that did."""

    name: str
    filename: str
    lineno: int


def broken_stdlib_module(exc: BaseException) -> BrokenStdlibModule | None:
    """Return the stdlib module that failed to import, or ``None`` for an ordinary crash.

    Only the innermost module body on the traceback decides: a stdlib module raising while
    it is imported means a damaged interpreter; Omnigent or a dependency importing a module
    this platform lacks is still a bug in that code.
    """
    innermost: BrokenStdlibModule | None = None
    tb = exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        if frame.f_code.co_name == "<module>":
            name = frame.f_globals.get("__name__")
            innermost = None
            if isinstance(name, str) and name.partition(".")[0] in sys.stdlib_module_names:
                innermost = BrokenStdlibModule(name, frame.f_code.co_filename, tb.tb_lineno)
        tb = tb.tb_next
    return innermost


def render_broken_stdlib_notice(
    exc: BaseException, module: BrokenStdlibModule, stream: TextIO | None = None
) -> None:
    """Explain that the interpreter, not Omnigent, is broken, and how to recover."""
    out = stream if stream is not None else sys.stderr
    version = ".".join(str(part) for part in sys.version_info[:3])
    lines = [
        "",
        "Omnigent cannot run on this Python installation: its standard library is broken.",
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
    lines += [
        "",
        "This is a problem with the Python installation, not with Omnigent, so there is",
        "nothing to report as an Omnigent bug. Repair or reinstall Python, then run",
        "omnigent again.",
        "",
    ]
    out.write("\n".join(lines) + "\n")
    out.flush()


def exit_if_stdlib_broken(exc: BaseException) -> None:
    """Exit 1 with the notice when the interpreter is at fault; otherwise return."""
    module = broken_stdlib_module(exc)
    if module is None:
        return
    render_broken_stdlib_notice(exc, module)
    raise SystemExit(1)
