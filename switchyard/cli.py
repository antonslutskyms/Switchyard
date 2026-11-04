# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for the native Switchyard server."""

import os
import shutil
import sys
from pathlib import Path

from switchyard import __version__


def main() -> None:
    """Forward server arguments to the native binary, preferring a local build."""
    args = sys.argv[1:]
    if args == ["--version"]:
        print(f"switchyard {__version__}")
        return
    if not args or args in (["--help"], ["-h"]):
        print("Usage: switchyard serve --config PATH [server options]\n"
              "Run switchyard serve --help for server options.")
        return
    if args[0] != "serve":
        raise SystemExit("Expected 'serve'. Run switchyard --help for usage.")
    local = Path(__file__).resolve().parents[1] / "target" / "release" / "switchyard-server"
    binary = str(local) if local.is_file() else shutil.which("switchyard-server")
    if binary is None:
        raise SystemExit("Install switchyard-server or build it with cargo build --release -p switchyard-server.")
    os.execv(binary, [binary, *args[1:]])


if __name__ == "__main__":
    main()
