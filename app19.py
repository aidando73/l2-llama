import argparse
import os
import json
from typing import Tuple
import requests
from ansi import bold, red, green, yellow, blue, magenta, cyan
from dotenv import load_dotenv
from agent17_5 import run_agent, MODEL_ID
import shutil
import time
from llama_stack_client import LlamaStackClient


def main(issue_url: str):
    github_api_key = os.getenv("GITHUB_API_KEY")
    if not github_api_key:
        raise ValueError("GITHUB_API_KEY is not set in the environment variables")

    llama_stack_url = os.getenv("LLAMA_STACK_URL")
    if not llama_stack_url:
        raise ValueError("LLAMA_STACK_URL is not set in the environment variables")

    client = LlamaStackClient(base_url=llama_stack_url)

    models = client.models.list()
    if MODEL_ID not in [model.identifier for model in models]:
        raise ValueError(f"Model {MODEL_ID} not found in LlamaStack. Llama Stack Coding Agent only supports {MODEL_ID} at the moment.")

    issue = Issue(issue_url)
    print(
        f"Solving issue {'#' + str(issue.issue_number)} in {f'{issue.owner}/{issue.repo}'}"
    )
    print()

    response = requests.get(
        f"https://api.github.com/repos/{issue.owner}/{issue.repo}/issues/{issue.issue_number}",
        headers={"Authorization": f"Bearer {github_api_key}"},
    )
    issue_data = response.json()
    print(f"Title: {issue_data['title']}")
    print(f"Body: {issue_data['body']}")
    print()

    # Make sure the sandbox directory exists
    os.makedirs("sandbox", exist_ok=True)

    # git clone the repo
    # Check if repo already exists and remove it if it does
    if not os.path.exists(f"sandbox/{issue.repo}"):
        print("Cloning repo...")
        os.system(
            f"cd sandbox && git clone https://{github_api_key}@github.com/{issue.owner}/{issue.repo}.git"
        )
        default_branch = (
            os.popen(
                f"cd sandbox/{issue.repo} && git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'"
            )
            .read()
            .strip()
        )
    else:
        default_branch = (
            os.popen(
                f"cd sandbox/{issue.repo} && git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'"
            )
            .read()
            .strip()
        )
        # Force checkout main branch if repo exists
        print("Cleaning up repo...")
        os.system(f"cd sandbox/{issue.repo} && git checkout -f {default_branch}")
        os.system(f"cd sandbox/{issue.repo} && git clean -fdx")

    # Run the agent
    agent_response = run_agent(client, issue.repo, issue_data["title"], issue_data["body"])

    branch_name = f"llama-agent-{issue.issue_number}-{int(time.time())}"

    changes_made = agent_response[0]
    if changes_made == "no_changes_made":
        reasoning = agent_response[1]

        os.system(f"touch sandbox/{issue.repo}/.keep")
        exit_code = os.system(
            f"cd sandbox/{issue.repo} && "
            f"git checkout -b {branch_name} && "
            f"git add . && "
            f"git commit -m 'Initial commit' && "
            f"git push origin {branch_name}"
        )

        # Create an issue comment explaining the reasoning
        response = requests.post(
            f"https://api.github.com/repos/{issue.owner}/{issue.repo}/pulls",
            headers={"Authorization": f"Bearer {github_api_key}"},
            json={
                "title": f"Agent attempted to solve: #{issue.issue_number} - {issue_data['title']}",
                "body": f"Agent attempted to resolve #{issue.issue_number}, but no changes were made. Here's it's explanation:\n\n{reasoning}",
                "head": branch_name,
                "base": default_branch,
            },
        )

        if response.status_code != 201:
            raise ValueError(f"Failed to create PR: {response.json()}")

        print()
        print(
            f"Agent attempted to solve the issue, but no changes were made. It's explanation is on the PR:\n\n"
            f"\t{yellow(response.json()['html_url'])}"
        )
    else:
        pr_title = agent_response[1]
        pr_body = agent_response[2]

        # Commit changes and create a new branch

        exit_code = os.system(
            f"cd sandbox/{issue.repo} && "
            f"git checkout -b {branch_name} && "
            f"git add . && "
            f"git commit -m 'Testing new PR' && "
            f"git push origin {branch_name}"
        )
        if exit_code != 0:
            raise ValueError("Failed to create new branch")

        # Create a new PR
        response = requests.post(
            f"https://api.github.com/repos/{issue.owner}/{issue.repo}/pulls",
            headers={"Authorization": f"Bearer {github_api_key}"},
            json={
                "title": f"#{issue.issue_number} - {pr_title}",
                "body": f"Resolves #{issue.issue_number}\n{pr_body}",
                "head": branch_name,
                "base": default_branch,
            },
        )
        if response.status_code != 201:
            raise ValueError(f"Failed to create new PR: {response.json()}")

        print()
        print(f"Created new PR: {green(response.json()['html_url'])}")


class Issue:
    repo: str
    owner: str
    issue_number: int

    def __init__(self, url: str):
        """
        Parse the issue url into the owner, repo and issue number
        """
        # Truncate the https:// prefix if it exists
        base_url = url
        if url.startswith("https://"):
            url = url[len("https://") :]

        # Check that the domain is github.com
        if not url.startswith("github.com"):
            raise ValueError(f"Expected github.com as the domain: {base_url}")

        parts = url.split("/")
        if (
            len(parts) < 5
        ):  # We expect 5 parts: github.com/owner/repo/issues/issue_number
            raise ValueError("Invalid GitHub issue URL format")

        self.owner = parts[1]
        self.repo = parts[2]
        if parts[3] != "issues":
            raise ValueError(f"Expected /issues/ in the URL: {base_url}")

        if parts[4] == "":
            raise ValueError(f"Expected an issue number in the URL: {base_url}")

        try:
            self.issue_number = int(parts[4])  # Issue number
        except ValueError:
            raise ValueError(f"Expected an integer issue number: {parts[4]}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--issue-url",
        type=str,
        required=True,
        help="The issue url to solve. E.g., https://github.com/aidando73/bitbucket-syntax-highlighting/issues/67",
    )
    args = parser.parse_args()

    main(issue_url=args.issue_url)
