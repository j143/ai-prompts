# Common Issues with AI Agents in Codebases

AI agents can be helpful but often introduce challenges. Below are common issues, solutions, and examples to address them.

## 1. Overwriting Code Instead of Making Line Changes
### Issue:
AI replaces entire sections of code, losing functionality.

### Solution:
- Use comments to indicate unchanged code.
- Validate changes with linters and tests.
- Use version control to revert mistakes.

#### Example:
```python
import os

def update_code(file_path, new_code):
    backup_path = f"{file_path}.backup"
    if os.path.exists(file_path):
        os.rename(file_path, backup_path)
    try:
        with open(file_path, 'w') as file:
            file.write(new_code)
    except Exception:
        os.rename(backup_path, file_path)
```

#### Decision Tree:
```mermaid
graph TD
    A[Start] --> B{Is the file critical?}
    B -->|Yes| C[Create a backup]
    B -->|No| D[Proceed with changes]
    C --> E[Apply changes]
    D --> E
    E --> F{Changes successful?}
    F -->|Yes| G[Finish]
    F -->|No| H[Restore from backup]
    H --> G
```

#### Wrong Way:
```python
# Overwrites the entire file without backup
with open("config.py", 'w') as file:
    file.write("NEW CONFIG")
```

#### Right Way:
```python
# Creates a backup before making changes
import os

def update_code(file_path, new_code):
    backup_path = f"{file_path}.backup"
    if os.path.exists(file_path):
        os.rename(file_path, backup_path)
    with open(file_path, 'w') as file:
        file.write(new_code)
```

## 2. Creating New Code Instead of Modifying Existing Code
### Issue:
AI duplicates functionality instead of reusing existing code.

### Solution:
- Search for existing implementations.
- Clearly specify where changes are needed.

#### Example:
```python
import os

def find_existing(keyword, directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and keyword in open(os.path.join(root, file)).read():
                return file
```

#### Decision Tree:
```mermaid
graph TD
    A[Start] --> B{Search for existing functionality}
    B -->|Found| C[Reuse existing code]
    B -->|Not Found| D[Create new code]
    C --> E[Integrate and test]
    D --> E
    E --> F[Finish]
```

#### Wrong Way:
```python
# Creates a duplicate function instead of reusing existing one
def calculate_sum(a, b):
    return a + b

def add_numbers(a, b):
    return a + b
```

#### Right Way:
```python
# Reuses the existing function
def calculate_sum(a, b):
    return a + b

# Use calculate_sum wherever needed
result = calculate_sum(2, 3)
```

## 3. Ignoring Dependencies or Context
### Issue:
AI removes imports or breaks interdependent modules.

### Solution:
- Analyze dependencies.
- Provide sufficient context.

#### Example:
```python
import subprocess

def install_dependencies(dependencies):
    for dep in dependencies:
        subprocess.run(["pip", "install", dep], check=True)
```

#### Decision Tree:
```mermaid
graph TD
    A[Start] --> B{Are dependencies installed?}
    B -->|Yes| C[Proceed with changes]
    B -->|No| D[Install dependencies]
    D --> C
    C --> E[Test functionality]
    E --> F[Finish]
```

#### Wrong Way:
```python
# Removes imports without checking usage
# import requests
response = requests.get("https://example.com")
```

#### Right Way:
```python
# Ensures all dependencies are installed and imported
import requests
response = requests.get("https://example.com")
```

## 4. Overcomplicating Simple Tasks
### Issue:
AI generates overly complex solutions.

### Solution:
- Simplify requirements.
- Refactor for clarity.

#### Example:
```python
# Simplified
result = sum(x**2 for x in range(10))
```

#### Decision Tree:
```mermaid
graph TD
    A[Start] --> B{Is the solution simple?}
    B -->|Yes| C[Proceed]
    B -->|No| D[Refactor to simplify]
    D --> C
    C --> E[Test and validate]
    E --> F[Finish]
```

#### Wrong Way:
```python
# Overcomplicated solution
result = 0
for x in range(10):
    result += x**2
```

#### Right Way:
```python
# Simplified solution
result = sum(x**2 for x in range(10))
```

## 5. Misinterpreting Instructions
### Issue:
AI misunderstands vague instructions.

### Solution:
- Provide clear, detailed instructions.

#### Example:
```python
# Clear instruction
"Fix the IndexError in calculate_sum when input is empty."
```

#### Decision Tree:
```mermaid
graph TD
    A[Start] --> B{Are instructions clear?}
    B -->|Yes| C[Proceed with task]
    B -->|No| D[Clarify instructions]
    D --> C
    C --> E[Test and validate]
    E --> F[Finish]
```

#### Wrong Way:
```python
# Misinterprets vague instructions
# "Fix the bug in the function."
def buggy_function():
    pass
```

#### Right Way:
```python
# Implements clear instructions
# "Fix the IndexError in calculate_sum when input is empty."
def calculate_sum(numbers):
    return sum(numbers) if numbers else 0
```

## 6. Lack of Testing
### Issue:
AI introduces changes without tests.

### Solution:
- Add tests for new functionality.

#### Example:
```python
def add(a, b):
    return a + b

assert add(2, 3) == 5
```

#### Decision Tree:
```mermaid
graph TD
    A[Start] --> B{Are tests written?}
    B -->|Yes| C[Run tests]
    B -->|No| D[Write tests]
    D --> C
    C --> E{Tests passed?}
    E -->|Yes| F[Finish]
    E -->|No| G[Fix issues and re-test]
    G --> C
```

#### Wrong Way:
```python
# No tests for new functionality
def add(a, b):
    return a + b
```

#### Right Way:
```python
# Adds tests for new functionality
def add(a, b):
    return a + b

assert add(2, 3) == 5
assert add(-1, 1) == 0
```

## 7. Ignoring Coding Standards
### Issue:
AI generates code that violates style guides.

### Solution:
- Use linters and enforce standards.

#### Example:
```python
# Compliant
x = 42

def foo():
    print("Hello")
```

#### Decision Tree:
```mermaid
graph TD
    A[Start] --> B{Run linter}
    B -->|No issues| C[Proceed]
    B -->|Issues found| D[Fix issues]
    D --> C
    C --> E[Finish]
```

#### Wrong Way:
```python
# Violates coding standards
x=  42

def foo ( ):
    print( "Hello" )
```

#### Right Way:
```python
# Adheres to coding standards
x = 42

def foo():
    print("Hello")
```

## 8. Overwriting or Deleting Critical Files
### Issue:
AI deletes or overwrites critical files.

### Solution:
- Backup files before changes.

#### Example:
```python
import os

def safe_delete(file_path):
    if os.path.exists(file_path):
        os.rename(file_path, f"{file_path}.backup")
```

#### Decision Tree:
```mermaid
graph TD
    A[Start] --> B{Is the file critical?}
    B -->|Yes| C[Create a backup]
    B -->|No| D[Delete file]
    C --> E[Proceed with caution]
    D --> E
    E --> F[Finish]
```

#### Wrong Way:
```python
# Deletes a file without backup
import os
os.remove("critical_config.yaml")
```

#### Right Way:
```python
# Creates a backup before deleting
import os

def safe_delete(file_path):
    if os.path.exists(file_path):
        os.rename(file_path, f"{file_path}.backup")
        os.remove(file_path)
```

## 9. Debugging AI-Generated Code
### Issue:
AI code lacks comments or uses unconventional patterns.

### Solution:
- Add comments and refactor for clarity.

#### Example:
```python
def calculate_area(radius):
    # Calculate the area of a circle
    return 3.14 * radius ** 2
```

#### Decision Tree:
```mermaid
graph TD
    A[Start] --> B{Is the code documented?}
    B -->|Yes| C[Proceed with debugging]
    B -->|No| D[Add comments and documentation]
    D --> C
    C --> E[Identify issues]
    E --> F[Fix and validate]
    F --> G[Finish]
```

#### Wrong Way:
```python
# No comments or documentation
def calculate_area(radius):
    return 3.14 * radius ** 2
```

#### Right Way:
```python
# Adds comments for clarity
def calculate_area(radius):
    # Calculate the area of a circle given its radius
    return 3.14 * radius ** 2
```

## 10. Lack of Workflow Integration
### Issue:
AI changes disrupt existing workflows.

### Solution:
- Align AI with CI/CD pipelines.

#### Example:
```python
import subprocess

def run_tests():
    result = subprocess.run(["pytest"], check=False)
    if result.returncode == 0:
        print("Tests passed.")
```

#### Decision Tree:
```mermaid
graph TD
    A[Start] --> B{Is the workflow integrated?}
    B -->|Yes| C[Proceed with task]
    B -->|No| D[Integrate with workflow]
    D --> C
    C --> E[Test and validate]
    E --> F[Finish]
```

#### Wrong Way:
```python
# Skips testing before deployment
def deploy():
    print("Deploying application...")
```

#### Right Way:
```python
# Integrates testing into the deployment process
import subprocess

def deploy():
    result = subprocess.run(["pytest"], check=False)
    if result.returncode == 0:
        print("Tests passed. Deploying application...")
    else:
        print("Tests failed. Aborting deployment.")
```
