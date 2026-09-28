"""Fetch Linear tickets whose status is Todo or Backlog.

Usage:
    cp .env.example .env    # then put your real key in .env
    uv run python -m workflow_test.fetch_tickets
"""

import json
import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv

LINEAR_API_URL = "https://api.linear.app/graphql"

# Linear groups every workflow state into a fixed "type".
# "Backlog" states have type "backlog"; "Todo" states have type "unstarted".
# Filtering on type (not name) still works if a team renames its states.
STATE_TYPES = ["backlog", "unstarted"]

QUERY = """
query Tickets($filter: IssueFilter, $after: String) {
  issues(first: 50, after: $after, filter: $filter) {
    nodes {
      identifier
      title
      url
      branchName
      description
      state { name type }
      assignee { name }
      team { key }
      attachments { nodes { title url } }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""


def run_query(api_key: str, variables: dict) -> dict:
    """Send one GraphQL request to Linear and return the "data" part."""
    body = json.dumps({"query": QUERY, "variables": variables}).encode()
    request = urllib.request.Request(
        LINEAR_API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            # Personal API keys go in the header as-is (no "Bearer" prefix).
            "Authorization": api_key,
        },
    )
    with urllib.request.urlopen(request) as response:
        payload = json.load(response)

    # GraphQL reports errors in the body, often with HTTP 200.
    if "errors" in payload:
        raise RuntimeError(payload["errors"])
    return payload["data"]


def fetch_tickets(api_key: str, assignee_email: str) -> list[dict]:
    """Return one person's issues in Todo or Backlog, following pagination."""
    # Top-level filter keys are combined with AND.
    # Match on email, not name: names can repeat, emails are unique.
    issue_filter = {
        "state": {"type": {"in": STATE_TYPES}},
        "assignee": {"email": {"eq": assignee_email}},
    }
    tickets: list[dict] = []
    cursor = None

    while True:
        data = run_query(api_key, {"filter": issue_filter, "after": cursor})
        page = data["issues"]
        tickets.extend(page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            return tickets
        cursor = page["pageInfo"]["endCursor"]


def main() -> None:
    # Copy KEY=value lines from .env into os.environ.
    # Variables already set in the shell win over .env values.
    load_dotenv()
    api_key = os.environ.get("LINEAR_API_KEY")
    if not api_key:
        sys.exit("Set LINEAR_API_KEY first (Linear > Settings > Security & access > API keys).")
    assignee_email = os.environ.get("LINEAR_ASSIGNEE_EMAIL")
    if not assignee_email:
        sys.exit("Set LINEAR_ASSIGNEE_EMAIL first (email of the Linear user whose tickets you want).")

    try:
        tickets = fetch_tickets(api_key, assignee_email)
    except urllib.error.HTTPError as error:
        sys.exit(f"HTTP {error.code}: {error.read().decode()}")

    for ticket in tickets:
        print_ticket(ticket)
    print(f"{len(tickets)} tickets")


def print_ticket(ticket: dict) -> None:
    assignee = ticket["assignee"]["name"] if ticket["assignee"] else "unassigned"
    print(f"{ticket['identifier']} [{ticket['state']['name']}] {ticket['title']} ({assignee})")
    print(f"  URL: {ticket['url']}")
    # The git branch name Linear suggests ("Copy git branch name" in the UI).
    print(f"  Branch: {ticket['branchName']}")

    # Description is Markdown, and is null when the ticket has none.
    description = (ticket["description"] or "").strip()
    print("  Description:")
    for line in (description or "(none)").splitlines():
        print(f"    {line}")

    # Attachments are the "Links" section in Linear: PRs, docs, Slack threads.
    links = ticket["attachments"]["nodes"]
    if links:
        print("  Links:")
        for link in links:
            print(f"    - {link['title']}: {link['url']}")
    print()


if __name__ == "__main__":
    main()
