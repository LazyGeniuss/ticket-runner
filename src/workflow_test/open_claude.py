"""Open an interactive Claude Code session in the AppraisalBureau directory.

Usage:
    uv run python -m workflow_test.open_claude
    uv run python -m workflow_test.open_claude "Summarise ab-api"   # optional first prompt
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# This file lives at AppraisalBureau/workflow-test/src/workflow_test/open_claude.py,
# so three levels up from its folder is AppraisalBureau.
DEFAULT_WORKDIR = Path(__file__).resolve().parents[3]


def open_claude(workdir: Path, prompt: str | None = None) -> int:
    """Start `claude` in workdir and wait until the user exits it."""
    claude = shutil.which("claude")
    if claude is None:
        sys.exit("`claude` not found on PATH. Install: npm install -g @anthropic-ai/claude-code")
    if not workdir.is_dir():
        sys.exit(f"Directory not found: {workdir}")

    command = [claude]
    if prompt:
        command.append(prompt)

    # cwd sets the folder Claude starts in. stdin/stdout are inherited,
    # so Claude takes over this terminal until you quit it.
    return subprocess.run(command, cwd=workdir).returncode


def main() -> None:
    load_dotenv()
    # Set CLAUDE_WORKDIR in .env to start Claude somewhere else.
    workdir = Path(os.environ.get("CLAUDE_WORKDIR", DEFAULT_WORKDIR))
    prompt = " ".join(sys.argv[1:]) or None
    sys.exit(open_claude(workdir, prompt))


if __name__ == "__main__":
    main()
