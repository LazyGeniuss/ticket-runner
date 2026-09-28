"""Route a task to backend, frontend or both, then run one subagent per area.

Step 1 (router): a read-only agent looks at the task and the repos, and
returns JSON saying which area the task belongs to.
Step 2 (workers): one agent per chosen area runs inside that repo, backend
first, then frontend. Each worker first writes a plan without changing files.
You review the plan, answer any questions, and approve. The same worker
session then implements it. The frontend worker is told what the backend
worker did, so it can match the API change.

Usage:
    uv run python -m workflow_test.agent "Add a status filter to the orders list"
"""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    ToolUseBlock,
    query,
)
from dotenv import load_dotenv

load_dotenv()

# Set AGENT_MODEL in .env to use another model, e.g. a local one through Ollama.
# The SDK runs the Claude Code CLI, which reads ANTHROPIC_BASE_URL from the
# environment, so pointing that at Ollama sends every request to it.
MODEL = os.environ.get("AGENT_MODEL", "claude-opus-5")
DEFAULT_WORKDIR = Path(__file__).resolve().parents[3]

# One entry per area, in the order the workers run. "repo" is the folder
# the worker starts in.
AREAS = {
    "backend": {
        "repo": "ab-api",
        "prompt": (
            "You are a backend engineer working on ab-api, a Node.js/TypeScript API "
            "backed by Strapi. Only change code in this repo. Follow the existing "
            "patterns for routes, services, database access and tests."
        ),
        "dev": ["npm", "run", "develop"],  # Strapi dev server
        "port": 1337,
        "url": "http://localhost:1337/admin",
    },
    "frontend": {
        "repo": "ab-frontend-next",
        "prompt": (
            "You are a frontend engineer working on ab-frontend-next, a Next.js 15 / "
            "React 18 app. Only change code in this repo. Reuse existing "
            "components, hooks and API clients before adding new ones."
        ),
        "dev": ["npm", "run", "dev"],  # Next dev server
        "port": 3000,
        "url": "http://localhost:3000",
    },
}

ROUTER_PROMPT = """\
You route engineering tasks for the Appraisal Bureau codebase.
The working directory holds two repos:
- ab-api: backend (Node.js/TypeScript API, PostgreSQL)
- ab-frontend-next: frontend (Next.js/React)

Decide which repo(s) the task needs changes in. Look at the code if the task
text alone is not enough. Choose "both" only when each repo needs its own change,
for example a new API field that the UI must also show.

For each repo you choose, write a self-contained task for an engineer who
works in only that repo and cannot see the other one. Say WHAT must change
and why, and point to the relevant files or endpoints you found. Do not decide
HOW: no helper functions, formats, component names or other implementation
choices the user did not ask for. The engineer will make those choices."""

# Structured output: the router's final answer must match this JSON schema.
ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "area": {"type": "string", "enum": ["backend", "frontend", "both"]},
        "reason": {"type": "string"},
        "backend_task": {"type": "string"},
        "frontend_task": {"type": "string"},
    },
    "required": ["area", "reason"],
}

ROUTER_TOOLS = ["Read", "Glob", "Grep"]

# Added to every worker's system prompt. The user only sees the worker's
# final message, so questions must be answerable from that message alone.
WORKER_RULES = """

The user reviews your work only through your final message. If you need them
to decide something, list every option in that message: a letter or number,
a one-line description, and your recommendation. Never refer to options only
by name (for example "Option A vs B") without describing them."""


def describe(block: ToolUseBlock) -> str:
    """One-line summary of a tool call, showing what it acts on."""
    args = block.input
    detail = args.get("command") or args.get("file_path") or args.get("pattern") or ""
    print(block)
    return f"{block.name} {detail}".strip()


async def run_agent(
    label: str,
    prompt: str,
    options: ClaudeAgentOptions,
    written_files: list[str] | None = None,
) -> ResultMessage:
    """Run one agent to the end, printing its tool calls as progress.

    If written_files is given, the path of every file the agent writes or
    edits is appended to it.
    """
    result = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    print(f"[{label}] {describe(block)}")
                    if written_files is not None and block.name in ("Write", "Edit"):
                        written_files.append(block.input.get("file_path", ""))
        elif isinstance(message, ResultMessage):
            result = message
    if result is None or result.is_error:
        raise RuntimeError(f"{label} agent failed: {result.result if result else 'no result'}")
    return result


async def route(task: str, workdir: Path) -> dict:
    """Step 1: decide which area(s) the task belongs to."""
    options = ClaudeAgentOptions(
        model=MODEL,
        cwd=workdir,
        system_prompt=ROUTER_PROMPT,
        # `tools` limits which tools exist at all. `allowed_tools` only
        # auto-approves them; on its own it would leave Agent, Bash etc. available.
        tools=ROUTER_TOOLS,
        allowed_tools=ROUTER_TOOLS,
        output_format={"type": "json_schema", "schema": ROUTE_SCHEMA},
        max_turns=15,
    )
    result = await run_agent("router", task, options)
    if result.structured_output is not None:
        return result.structured_output
    # Small local models sometimes ignore output_format and reply with plain
    # JSON text instead. Accept that, and fail clearly on anything else.
    try:
        return json.loads(result.result)
    except (TypeError, json.JSONDecodeError):
        raise RuntimeError(f"Router did not return JSON: {result.result!r}")


def worker_options(area: str, workdir: Path, **extra) -> ClaudeAgentOptions:
    config = AREAS[area]
    return ClaudeAgentOptions(
        model=MODEL,
        cwd=workdir / config["repo"],
        # Keep Claude Code's built-in prompt (tool use, coding habits) and add the role.
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": config["prompt"] + WORKER_RULES,
        },
        max_turns=50,
        **extra,
    )


def ask_approval(area: str) -> str | None:
    """Ask the user to approve a plan. Returns the reply, or None to skip."""
    print(f"\nApprove the {area} plan?")
    print("  y       implement it as written")
    print("  <text>  implement it using your answers or changes, e.g. 'use formatDate'")
    print("  n       skip this area")
    reply = input("> ").strip()
    if reply.lower() in ("", "n", "no"):
        return None
    return "" if reply.lower() in ("y", "yes") else reply


BASE_BRANCH = "main"


class BranchError(Exception):
    pass


def git(repo_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo_path, capture_output=True, text=True)


def prepare_branch(repo_path: Path, branch: str) -> None:
    """Switch repo_path to `branch`, creating it from the latest main if needed.

    Done in Python, not by the agent: git steps must happen the same way every
    time, and never touch uncommitted work.
    """
    # Refuse to switch branches over uncommitted changes, so nothing is lost
    # or carried onto the new branch by accident.
    status = git(repo_path, "status", "--porcelain")
    if status.returncode != 0:
        raise BranchError(f"not a git repo: {status.stderr.strip()}")
    if status.stdout.strip():
        current = git(repo_path, "branch", "--show-current").stdout.strip()
        raise BranchError(
            f"uncommitted changes on '{current}'. Commit or stash them first:\n{status.stdout}"
        )

    # Branch already exists (for example, a second run on the same ticket):
    # switch to it and keep its work. `git switch` also picks up a branch
    # that exists only on origin.
    local = git(repo_path, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")
    remote = git(repo_path, "rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}")
    if local.returncode == 0 or remote.returncode == 0:
        switched = git(repo_path, "switch", branch)
        if switched.returncode != 0:
            raise BranchError(switched.stderr.strip())
        print(f"Switched to existing branch {branch}")
        return

    # New branch: start from the latest main on origin, or local main if
    # fetching fails (for example, no network).
    base = f"origin/{BASE_BRANCH}"
    fetched = git(repo_path, "fetch", "origin", BASE_BRANCH)
    if fetched.returncode != 0:
        print(f"Warning: git fetch failed, using local {BASE_BRANCH}: {fetched.stderr.strip()}")
        base = BASE_BRANCH
    created = git(repo_path, "switch", "--no-track", "-c", branch, base)
    if created.returncode != 0:
        raise BranchError(created.stderr.strip())
    print(f"Created branch {branch} from {base}")


@dataclass
class WorkerResult:
    """What one worker did, kept so we can resume it for test feedback."""

    area: str
    repo: str
    session_id: str
    summary: str


async def run_worker(area: str, task: str, workdir: Path, branch: str | None = None) -> WorkerResult | None:
    """Step 2: plan, get approval, then implement in the same session.

    If branch is given, the repo is switched to it (created from main if new)
    before the worker starts. Returns a WorkerResult, or None if skipped.
    """
    repo = AREAS[area]["repo"]

    if branch:
        try:
            prepare_branch(workdir / repo, branch)
        except BranchError as error:
            print(f"Skipped {area}: cannot switch {repo} to {branch}: {error}")
            return None
        task += f"\n\nYou are on git branch {branch}, created from {BASE_BRANCH} for this task."

    # Plan mode: the worker can read code and write a plan file, not edit code.
    written: list[str] = []
    plan = await run_agent(
        area, task, worker_options(area, workdir, permission_mode="plan"), written
    )

    # The full plan, including any options the worker asks you to choose
    # between, lives in the plan file. The final message is often only a summary.
    plan_files = [p for p in written if "/.claude/plans/" in p and Path(p).is_file()]
    if plan_files:
        print(f"\n===== {area} plan file: {plan_files[-1]} =====")
        print(Path(plan_files[-1]).read_text())
    print(f"\n===== {area} plan summary ({repo}) =====\n{plan.result}")

    reply = ask_approval(area)
    if reply is None:
        print(f"Skipped {area}.")
        return None

    # resume=session_id continues the same conversation, so the worker still
    # has its plan and everything it read. acceptEdits lets it change files.
    instruction = "The plan is approved. Implement it now."
    if reply:
        instruction += f"\n\nThe user's answers and changes to the plan:\n{reply}"
    instruction += "\n\nWhen done, summarise exactly what you changed."
    done = await run_agent(
        area,
        instruction,
        worker_options(area, workdir, permission_mode="acceptEdits", resume=plan.session_id),
    )
    print(f"\n===== {area} done ({repo}) =====\n{done.result}")
    return WorkerResult(area=area, repo=repo, session_id=done.session_id, summary=done.result)


async def apply_feedback(result: WorkerResult, workdir: Path, feedback: str) -> None:
    """Resume a worker with the user's test feedback so it makes more edits."""
    instruction = (
        "The user tested the running app and asked for these changes:\n"
        f"{feedback}\n\nMake them, then summarise exactly what you changed."
    )
    done = await run_agent(
        result.area,
        instruction,
        worker_options(result.area, workdir, permission_mode="acceptEdits", resume=result.session_id),
    )
    # Keep the newest session id so the next round of feedback resumes from here.
    result.session_id = done.session_id
    result.summary = done.result
    print(f"\n===== {result.area} updated ({result.repo}) =====\n{done.result}")


def wait_for_port(port: int, timeout: float = 180.0) -> bool:
    """Return True once something is listening on localhost:port."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            sock.settimeout(1.0)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1.0)
    return False


def start_dev_server(area: str, repo_path: Path) -> tuple[subprocess.Popen, Path]:
    """Start the repo's dev server as a background process, logging to a file."""
    cfg = AREAS[area]
    log_path = Path(tempfile.gettempdir()) / f"agent-dev-{area}.log"
    log = open(log_path, "w")
    # start_new_session so we can stop the whole process group later; Strapi and
    # Next each spawn child processes that a plain terminate() would leave running.
    proc = subprocess.Popen(
        cfg["dev"], cwd=repo_path, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    return proc, log_path


def stop_dev_server(proc: subprocess.Popen) -> None:
    import signal

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


async def local_test_loop(results: list[WorkerResult], workdir: Path) -> bool:
    """Run each changed repo locally, let the user test and request changes.

    Returns True if the user approved everything, False if they cancelled.
    """
    servers: dict[str, subprocess.Popen] = {}
    by_area = {r.area: r for r in results}
    try:
        for result in results:
            cfg = AREAS[result.area]
            proc, log_path = start_dev_server(result.area, workdir / result.repo)
            servers[result.area] = proc
            print(f"\nStarting {result.repo} ({' '.join(cfg['dev'])})... log: {log_path}")
            if wait_for_port(cfg["port"]):
                print(f"  {result.area} ready at {cfg['url']}")
            else:
                print(f"  {result.area} did not open port {cfg['port']} in time. Check the log.")

        while True:
            print("\nTest the app in your browser:")
            for result in results:
                print(f"  {result.area}: {AREAS[result.area]['url']}")
            print(
                "\nType 'ok' to approve everything, '<area>: what to change' to request "
                "edits\n(e.g. 'frontend: make the button blue'), or 'stop' to cancel."
            )
            reply = input("> ").strip()
            low = reply.lower()
            if low in ("ok", "y", "yes", "approve"):
                return True
            if low in ("stop", "cancel", "n", "no", "q"):
                return False
            if ":" in reply:
                area, feedback = (part.strip() for part in reply.split(":", 1))
                if area in by_area and feedback:
                    await apply_feedback(by_area[area], workdir, feedback)
                    print("Change applied. The dev server reloads automatically; test again.")
                    continue
            print(f"Use 'ok', 'stop', or one of: {', '.join(by_area)}: <what to change>.")
    finally:
        for proc in servers.values():
            stop_dev_server(proc)
        print("\nStopped the local servers.")


def push_to_github(results: list[WorkerResult], workdir: Path, branch: str, subject: str) -> None:
    """Ask per repo, then commit any changes and push the branch to origin."""
    for result in results:
        answer = input(f"\nPush {result.repo} (branch {branch}) to GitHub? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print(f"Did not push {result.repo}.")
            continue
        repo_path = workdir / result.repo
        # Commit whatever the worker changed. Skip the commit if nothing is staged.
        git(repo_path, "add", "-A")
        staged = git(repo_path, "diff", "--cached", "--quiet")
        if staged.returncode != 0:  # non-zero means there are staged changes
            committed = git(repo_path, "commit", "-m", subject)
            if committed.returncode != 0:
                print(f"  commit failed: {committed.stderr.strip()}")
                continue
        pushed = git(repo_path, "push", "-u", "origin", branch)
        if pushed.returncode == 0:
            print(f"  pushed {result.repo} to origin/{branch}")
        else:
            print(f"  push failed: {pushed.stderr.strip()}")


async def main_async(task: str, workdir: Path, branch: str | None = None) -> list[str]:
    """Route and run the task. Returns the areas that were implemented.

    If branch is given, each chosen repo gets that branch before its worker runs.
    """
    decision = await route(task, workdir)
    print(f"\nRoute: {decision['area']} ({decision['reason']})\n")

    areas = list(AREAS) if decision["area"] == "both" else [decision["area"]]
    backend_summary = None
    results: list[WorkerResult] = []

    # Run workers one after another (backend first) so the frontend worker
    # can build on the API change the backend worker actually made.
    for area in areas:
        # Fall back to the original task if the router left a sub-task empty.
        area_task = decision.get(f"{area}_task") or task
        if area == "frontend" and backend_summary:
            area_task += (
                "\n\nA backend engineer has already changed ab-api for this task. "
                f"Their summary:\n{backend_summary}"
            )
        result = await run_worker(area, area_task, workdir, branch)
        if result is not None:
            results.append(result)
            if area == "backend":
                backend_summary = result.summary

    if not results:
        return []

    # Let the user test the changes locally and request more edits.
    approved = await local_test_loop(results, workdir)
    if not approved:
        print("Cancelled. Nothing was pushed; your changes stay on the branch.")
    elif branch:
        # First line of the task is a good commit subject (for a ticket it is
        # "Linear ticket APP-XXXX: title").
        push_to_github(results, workdir, branch, task.splitlines()[0])
    else:
        print("Approved. No --branch given, so nothing was pushed.")

    return [r.area for r in results]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", help="What needs to be done, in plain words")
    parser.add_argument("--branch", help="Git branch to create from main and work on")
    args = parser.parse_args()

    workdir = Path(os.environ.get("CLAUDE_WORKDIR", DEFAULT_WORKDIR))
    asyncio.run(main_async(args.task, workdir, args.branch))


if __name__ == "__main__":
    main()
