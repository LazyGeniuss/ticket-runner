"""Pick a Linear ticket from a list, let the agent solve it, then pick again.

Usage:
    uv run python -m workflow_test.solve_tickets

Loop:
1. Fetch your Todo/Backlog tickets from Linear and show them as a numbered list.
2. You pick one by number (or q to quit).
3. The agent routes the ticket. For each chosen repo it creates Linear's
   suggested branch from main, then runs that repo's worker, asking you to
   approve each plan (see agent.py).
4. Back to step 1. The list is fetched again, so it stays up to date.

Tickets solved in this run are marked "done" in the list. Their status in
Linear does not change.
"""

import asyncio
import os
import sys
import urllib.error
from pathlib import Path

from workflow_test.agent import DEFAULT_WORKDIR, main_async
from workflow_test.fetch_tickets import fetch_tickets, print_ticket


def ticket_to_task(ticket: dict) -> str:
    """Turn a Linear ticket into the task text the agent receives."""
    parts = [f"Linear ticket {ticket['identifier']}: {ticket['title']}", ticket["url"]]
    description = (ticket["description"] or "").strip()
    if description:
        parts.append(f"Description:\n{description}")
    links = ticket["attachments"]["nodes"]
    if links:
        parts.append("Links:\n" + "\n".join(f"- {l['title']}: {l['url']}" for l in links))
    return "\n\n".join(parts)


def choose_ticket(tickets: list[dict], solved: set[str]) -> dict | None:
    """Show a numbered list and return the chosen ticket, or None to quit."""
    print("\nYour Todo/Backlog tickets:")
    for number, ticket in enumerate(tickets, start=1):
        mark = "  (done)" if ticket["identifier"] in solved else ""
        print(f"  {number:>2}. {ticket['identifier']:<10} [{ticket['state']['name']}] {ticket['title']}{mark}")

    while True:
        choice = input("\nPick a ticket number (q to quit): ").strip().lower()
        if choice in ("q", "quit", "exit"):
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(tickets):
            return tickets[int(choice) - 1]
        print(f"Enter a number from 1 to {len(tickets)}, or q.")


def main() -> None:
    # agent.py already loaded .env when it was imported.
    api_key = os.environ.get("LINEAR_API_KEY")
    assignee_email = os.environ.get("LINEAR_ASSIGNEE_EMAIL")
    if not api_key or not assignee_email:
        sys.exit("Set LINEAR_API_KEY and LINEAR_ASSIGNEE_EMAIL in .env first.")
    workdir = Path(os.environ.get("CLAUDE_WORKDIR", DEFAULT_WORKDIR))

    solved: set[str] = set()
    while True:
        try:
            tickets = fetch_tickets(api_key, assignee_email)
        except urllib.error.HTTPError as error:
            sys.exit(f"Linear HTTP {error.code}: {error.read().decode()}")
        if not tickets:
            print("No Todo/Backlog tickets assigned to you.")
            return

        ticket = choose_ticket(tickets, solved)
        if ticket is None:
            return

        print()
        print_ticket(ticket)
        try:
            # Each chosen repo gets Linear's suggested branch, created from main.
            implemented = asyncio.run(
                main_async(ticket_to_task(ticket), workdir, ticket["branchName"])
            )
            # Only mark it done if at least one plan was approved and implemented.
            if implemented:
                solved.add(ticket["identifier"])
        except KeyboardInterrupt:
            # Ctrl+C stops the current ticket but keeps the loop running.
            print(f"\nStopped {ticket['identifier']}. Back to the list.")
        except Exception as error:
            # One failed ticket should not end the session.
            print(f"\n{ticket['identifier']} failed: {error}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
