import anthropic
   import os
   import argparse
   from github import Github
   import subprocess
   
   def execute_task(task_description, issue_number):
       # 1. Вызываем Claude для генерации кода
       client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
       
       # Читаем текущий код
       relevant_files = ["main.py", "config.py", "db.py"]  # TODO: парси из task
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
       
       # 2. Парсим ответ и применяем изменения
       changes = parse_claude_response(response.content[0].text)
       
       for filepath, content in changes.items():
           os.makedirs(os.path.dirname(filepath), exist_ok=True)
           with open(filepath, 'w') as f:
               f.write(content)
           subprocess.run(["git", "add", filepath])
       
       # 3. Создаём коммит и PR
       branch_name = f"claude-task-{issue_number}"
       subprocess.run(["git", "checkout", "-b", branch_name])
       subprocess.run(["git", "commit", "-m", f"Fix #{issue_number}: {task_description[:50]}"])
       subprocess.run(["git", "push", "origin", branch_name])
       
       # 4. Открываем PR
       g = Github(os.environ["GH_TOKEN"])
       repo = g.get_repo(os.environ["GITHUB_REPOSITORY"])
       pr = repo.create_pull(
           title=f"🤖 Claude: {task_description[:50]}",
           body=f"Автоматический PR от Claude для решения #{issue_number}\n\n{response.content[0].text}",
           head=branch_name,
           base="main"
       )
       
       # 5. Комментим в issue
       issue = repo.get_issue(issue_number)
       issue.create_comment(f"✅ Создал PR: {pr.html_url}")
   
   def parse_claude_response(text):
       """Парсит ответ Claude и извлекает файлы"""
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
