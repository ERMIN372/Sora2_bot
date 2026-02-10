import anthropic
import os
import argparse
from github import Github
from pathlib import Path
import subprocess

# Whitelist of directories that Claude is allowed to write to.
_ALLOWED_ROOTS = {
    Path.cwd(),
}

def _safe_resolve(filepath: str) -> Path:
    """Resolve *filepath* and ensure it stays within the allowed project tree."""
    resolved = Path(filepath).resolve()
    for root in _ALLOWED_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise ValueError(
        f"Path traversal blocked: {filepath!r} resolves outside allowed roots"
    )


def execute_task(task_description, issue_number):
    # 1. Call Claude to generate code
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Read current code
    relevant_files = ["main.py", "config.py", "db.py"]  # TODO: parse from task
    context = "\n\n".join([
        f"### {f}\n```python\n{open(f).read()}\n```"
        for f in relevant_files if os.path.exists(f)
    ])

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        messages=[{
            "role": "user",
            "content": f"""
Ты AI-помощник для рефакторинга Python проекта.

Текущий код:
{context}

Задание:
{task_description}

Выдай ТОЛЬКО изменённые файлы в формате:

### FILENAME: path/to/file.py
```python
# полный код файла
```

### FILENAME: another/file.py
```python
# полный код файла
```
            """
        }]
    )

    # 2. Parse response and apply changes
    changes = parse_claude_response(response.content[0].text)

    for filepath, content in changes.items():
        safe_path = _safe_resolve(filepath)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content)
        subprocess.run(["git", "add", str(safe_path)])

    # 3. Create commit and PR
    branch_name = f"claude-task-{issue_number}"
    subprocess.run(["git", "checkout", "-b", branch_name])
    subprocess.run(["git", "commit", "-m", f"Fix #{issue_number}: {task_description[:50]}"])
    subprocess.run(["git", "push", "origin", branch_name])

    # 4. Open PR
    g = Github(os.environ["GH_TOKEN"])
    repo = g.get_repo(os.environ["GITHUB_REPOSITORY"])
    pr = repo.create_pull(
        title=f"Claude: {task_description[:50]}",
        body=f"Automatic PR by Claude for #{issue_number}\n\n{response.content[0].text}",
        head=branch_name,
        base="main"
    )

    # 5. Comment on issue
    issue = repo.get_issue(issue_number)
    issue.create_comment(f"Created PR: {pr.html_url}")

def parse_claude_response(text):
    """Parse Claude's response and extract files."""
    import re
    files = {}
    pattern = r'### FILENAME: (.+?)\n```python\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for filepath, content in matches:
        files[filepath.strip()] = content.strip()
    return files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--issue-number", required=True, type=int)
    args = parser.parse_args()

    execute_task(args.task, args.issue_number)
