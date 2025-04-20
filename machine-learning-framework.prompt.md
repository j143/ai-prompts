# Building ML Frameworks with AI Assistance: Implementation Guide

## Project Structure for an AI-Assisted ML Framework

```
ml-framework/
├── .devcontainer/                    # Codespaces configuration
│   ├── devcontainer.json             # Dev container config
│   └── Dockerfile                    # Dev environment setup
├── .github/
│   ├── workflows/                    # CI/CD pipelines
│   │   ├── tests.yml                 # Test automation
│   │   └── docs-generation.yml       # Documentation CI
│   └── CODEOWNERS                    # Code ownership
├── src/
│   ├── core/                         # Core ML functionality
│   ├── models/                       # Model implementations
│   ├── preprocessing/                # Data preprocessing tools
│   ├── metrics/                      # Evaluation metrics
│   └── visualization/                # Visualization utilities
├── tests/
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   └── benchmark/                    # Performance benchmarks
├── examples/                         # Usage examples
├── docs/
│   ├── api/                          # API documentation
│   ├── _templates/                   # Doc templates
│   └── tutorials/                    # In-depth tutorials
├── notebooks/                        # Jupyter notebooks
└── .copilot/                         # AI assistance guides
    ├── prompts/                      # Specialized prompts
    └── examples/                     # Examples for AI
```

## Concrete Implementation Techniques

### 1. AI-Assisted Documentation Generation

#### Docstring Templates for AI Auto-Generation

```python
# src/models/linear_model.py

def ridge_regression(X, y, alpha=1.0, solver='auto', max_iter=None, tol=1e-3):
    """
    AI-DOC-TEMPLATE: Algorithm Implementation
    
    Implements ridge regression with support for multiple solvers.
    
    Parameters:
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data
    
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values
    
    alpha : float or array-like of shape (n_targets,), default=1.0
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        
    solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'}, default='auto'
        Solver to use in the computational routines.
        
    max_iter : int, default=None
        Maximum number of iterations for conjugate gradient solver.
        The default value is determined by scipy.sparse.linalg.
        
    tol : float, default=1e-3
        Precision of the solution.
        
    Returns:
    -------
    coef : ndarray of shape (n_features,) or (n_targets, n_features)
        Weight vector(s).
        
    intercept : float or ndarray of shape (n_targets,)
        Independent term in decision function.
        
    Examples:
    --------
    >>> import numpy as np
    >>> from ml_framework import ridge_regression
    >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    >>> y = np.array([0, 1, 1, 0])
    >>> coef, intercept = ridge_regression(X, y, alpha=0.5)
    >>> print(coef)
    [0.0, 0.0]
    
    Notes:
    -----
    This implementation uses the algorithm described in:
    Hastie, T., Tibshirani, R., Friedman, J. (2009). The Elements of
    Statistical Learning. Springer, New York. pp. 63-65.
    """
    # Implementation here...
```

#### Documentation Generation Script

```python
# scripts/generate_docs.py

import os
import re
import sys
from pathlib import Path
import openai
import argparse

def extract_docstrings(file_path):
    """Extract docstrings from Python files."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Regex to extract docstrings
    pattern = r'def\s+(\w+)\s*\(([^)]*)\):\s*"""(.*?)"""'
    matches = re.findall(pattern, content, re.DOTALL)
    
    docstrings = []
    for match in matches:
        function_name, parameters, docstring = match
        if 'AI-DOC-TEMPLATE' in docstring:
            docstrings.append({
                'function': function_name,
                'parameters': parameters,
                'docstring': docstring.strip(),
                'template_type': re.search(r'AI-DOC-TEMPLATE:\s*(.*)', docstring).group(1)
            })
    
    return docstrings

def enhance_documentation(docstring_info):
    """Use AI to enhance documentation based on templates."""
    # Initialize API client
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    enhanced_docs = []
    
    for doc in docstring_info:
        # Prepare prompt based on template type
        if doc['template_type'] == 'Algorithm Implementation':
            prompt = f"""
            Based on this function signature and docstring, please enhance the documentation with:
            1. More detailed explanation of the algorithm
            2. Mathematical formula representation (in LaTeX)
            3. Extended example with more realistic data
            4. Additional usage notes and warnings
            5. References to relevant scientific papers
            
            Function: {doc['function']}
            Parameters: {doc['parameters']}
            Current docstring: {doc['docstring']}
            """
        elif doc['template_type'] == 'Data Processor':
            # Different prompt for data processors
            pass  # Other template types...
        
        # Call AI to enhance documentation
        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=1000
        )
        
        enhanced_docs.append({
            'function': doc['function'],
            'original': doc['docstring'],
            'enhanced': response.choices[0].text.strip()
        })
    
    return enhanced_docs

def main():
    parser = argparse.ArgumentParser(description='Generate enhanced documentation for ML framework')
    parser.add_argument('--source', required=True, help='Source directory to scan')
    parser.add_argument('--output', required=True, help='Output directory for enhanced docs')
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Process all Python files
    for py_file in source_dir.glob('**/*.py'):
        relative_path = py_file.relative_to(source_dir)
        print(f"Processing {relative_path}...")
        
        docstrings = extract_docstrings(py_file)
        if docstrings:
            enhanced = enhance_documentation(docstrings)
            
            # Write enhanced documentation
            doc_output = output_dir / relative_path.with_suffix('.md')
            doc_output.parent.mkdir(exist_ok=True, parents=True)
            
            with open(doc_output, 'w') as f:
                f.write(f"# {relative_path}\n\n")
                for doc in enhanced:
                    f.write(f"## `{doc['function']}`\n\n")
                    f.write(doc['enhanced'])
                    f.write("\n\n---\n\n")

if __name__ == "__main__":
    main()
```

### 2. AI-Assisted Test Creation

#### Test Template Generation System

```python
# scripts/generate_tests.py

import ast
import os
import sys
from pathlib import Path
import openai
import argparse
import importlib.util

def extract_function_info(file_path):
    """Extract function signatures and docstrings for test generation."""
    with open(file_path, 'r') as f:
        source_code = f.read()
    
    tree = ast.parse(source_code)
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip private functions
            if node.name.startswith('_') and not node.name.startswith('__'):
                continue
                
            # Extract docstring
            docstring = ast.get_docstring(node)
            
            # Extract parameters
            params = []
            for arg in node.args.args:
                params.append(arg.arg)
            
            # Extract return annotation if present
            return_type = None
            if node.returns:
                return_type = ast.unparse(node.returns)
            
            functions.append({
                'name': node.name,
                'params': params,
                'docstring': docstring,
                'return_type': return_type,
                'line_number': node.lineno
            })
    
    return functions

def generate_test_cases(module_path, function_info):
    """Generate test cases using AI for the given function."""
    # Prepare imports for dynamic loading
    module_name = os.path.basename(module_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get actual function
    function = getattr(module, function_info['name'])
    
    # Prepare prompt for test generation
    params_str = ', '.join(function_info['params'])
    return_type_str = function_info['return_type'] if function_info['return_type'] else "Not specified"
    
    prompt = f"""
    Generate pytest unit tests for the following function:
    
    Function name: {function_info['name']}
    Parameters: {params_str}
    Return type: {return_type_str}
    Docstring: {function_info['docstring']}
    
    Create test cases that cover:
    1. Normal usage with expected inputs
    2. Edge cases (empty inputs, boundary values)
    3. Error cases (invalid inputs, type errors)
    4. Special cases mentioned in the docstring
    
    For each test:
    - Include descriptive name
    - Add proper assertions
    - Add comments explaining the test purpose
    - Use pytest fixtures where appropriate
    
    Please format the response as valid Python test code that can be directly saved to a file.
    """
    
    # Call OpenAI API
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    response = openai.Completion.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=1500
    )
    
    return response.choices[0].text.strip()

def main():
    parser = argparse.ArgumentParser(description='Generate tests for ML framework modules')
    parser.add_argument('--source', required=True, help='Source module to generate tests for')
    parser.add_argument('--output', required=True, help='Output directory for test files')
    args = parser.parse_args()
    
    source_path = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if source_path.is_file():
        # Process single file
        module_path = source_path
        functions = extract_function_info(module_path)
        
        # Generate test file
        test_file_name = f"test_{module_path.stem}.py"
        test_file_path = output_dir / test_file_name
        
        with open(test_file_path, 'w') as f:
            f.write("import pytest\n")
            f.write(f"from {module_path.stem} import *\n\n")
            
            for func_info in functions:
                print(f"Generating tests for {func_info['name']}...")
                test_code = generate_test_cases(module_path, func_info)
                f.write(f"# Tests for {func_info['name']}\n")
                f.write(test_code)
                f.write("\n\n")
        
        print(f"Tests written to {test_file_path}")
    
    else:
        # Process directory
        for py_file in source_path.glob('**/*.py'):
            if py_file.name.startswith('test_'):
                continue  # Skip existing test files
                
            relative_path = py_file.relative_to(source_path)
            test_file_path = output_dir / f"test_{relative_path}"
            test_file_path.parent.mkdir(exist_ok=True, parents=True)
            
            functions = extract_function_info(py_file)
            
            with open(test_file_path, 'w') as f:
                f.write("import pytest\n")
                f.write(f"from {py_file.stem} import *\n\n")
                
                for func_info in functions:
                    print(f"Generating tests for {relative_path}:{func_info['name']}...")
                    test_code = generate_test_cases(py_file, func_info)
                    f.write(f"# Tests for {func_info['name']}\n")
                    f.write(test_code)
                    f.write("\n\n")
            
            print(f"Tests written to {test_file_path}")

if __name__ == "__main__":
    main()
```

### 3. AI Code Review Automation

Create custom GitHub Actions workflow for ML-specific code review:

```yaml
# .github/workflows/ai-code-review.yml
name: AI Code Review

on:
  pull_request:
    types: [opened, synchronize]
    paths:
      - 'src/**/*.py'
      - 'tests/**/*.py'

jobs:
  ai-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
      
      - name: Get changed files
        id: changed-files
        uses: tj-actions/changed-files@v37
        with:
          files: |
            src/**/*.py
            tests/**/*.py
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install openai pylint
      
      - name: Run AI code review
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          echo "${{ steps.changed-files.outputs.all_changed_files }}" > changed_files.txt
          python .github/scripts/ai_code_review.py
```

Implement the AI review script:

```python
# .github/scripts/ai_code_review.py
import os
import re
import sys
import json
import openai
import subprocess
from github import Github

def get_diff(file_path):
    """Get the git diff for the specified file."""
    cmd = ["git", "diff", "origin/main", "--", file_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def run_pylint(file_path):
    """Run pylint on the file and return issues."""
    cmd = ["pylint", "--output-format=json", file_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

def analyze_code(file_path, diff, pylint_issues):
    """Use AI to analyze code and suggest improvements."""
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    # Get file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Prepare ML-specific review prompt
    prompt = f"""
    You are a senior machine learning engineer reviewing code for an ML framework.
    Please review the following Python file content and diff, focusing on:
    
    1. ML-specific best practices
    2. Numerical stability issues
    3. Potential performance bottlenecks
    4. Algorithm correctness
    5. Input validation for ML contexts
    6. Documentation completeness for ML functions
    
    Pylint issues detected: {json.dumps(pylint_issues)}
    
    File content:
    ```python
    {content}
    ```
    
    Diff from main branch:
    ```diff
    {diff}
    ```
    
    Provide your review as a concise markdown list with specific recommendations,
    focusing on important ML-related issues rather than stylistic concerns.
    For each issue, include the line number and suggested fix where applicable.
    """
    
    response = openai.Completion.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=1000
    )
    
    return response.choices[0].text.strip()

def post_review_comment(repo_name, pr_number, review_comments):
    """Post the AI review as a comment on the PR."""
    g = Github(os.environ.get("GITHUB_TOKEN"))
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    
    comment = f"""
    # AI Code Review 🤖
    
    {review_comments}
    
    ---
    *This review was automatically generated by the ML Framework's AI code review system.*
    """
    
    pr.create_issue_comment(comment)

def main():
    # Get PR details from GitHub environment
    github_repository = os.environ.get("GITHUB_REPOSITORY")
    pr_number = int(os.environ.get("GITHUB_EVENT_PATH").split('/')[-1].split('.')[0])
    
    # Read changed files
    with open("changed_files.txt", "r") as f:
        changed_files = f.read().splitlines()
    
    all_reviews = []
    
    for file_path in changed_files:
        print(f"Reviewing {file_path}...")
        diff = get_diff(file_path)
        pylint_issues = run_pylint(file_path)
        review = analyze_code(file_path, diff, pylint_issues)
        
        all_reviews.append(f"## {file_path}\n\n{review}\n")
    
    # Post the combined review
    post_review_comment(github_repository, pr_number, "\n".join(all_reviews))
    print("Review posted successfully!")

if __name__ == "__main__":
    main()
```

### 4. Interactive Architecture Design with AI

Create a `.devcontainer.json` that supports AI-powered ML framework development:

```json
// .devcontainer/devcontainer.json
{
  "name": "ML Framework Development",
  "dockerFile": "Dockerfile",
  "settings": {
    "python.pythonPath": "/usr/local/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.rulers": [88],
    "jupyter.alwaysTrustNotebooks": true
  },
  "extensions": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",
    "github.copilot",
    "github.copilot-chat",
    "njpwerner.autodocstring"
  ],
  "forwardPorts": [8888],
  "postCreateCommand": "pip install -e '.[dev]'",
  "remoteUser": "vscode",
  "features": {
    "github-cli": "latest"
  }
}
```

Create AI guidance templates in `.copilot` directory:

```markdown
# .copilot/prompts/algorithm_implementation.md

# ML Algorithm Implementation Guide

When implementing ML algorithms, follow this pattern to ensure AI assistance works effectively:

## 1. Function Template Structure

```python
def algorithm_name(X, y, **hyperparameters):
    """
    AI-DOC-TEMPLATE: Algorithm Implementation
    
    Brief description of what the algorithm does.
    
    Parameters:
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.
    
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    
    hyperparameter1 : type, default=value
        Description of the hyperparameter.
    
    Returns:
    -------
    return_value1 : type
        Description of return value.
    
    return_value2 : type
        Description of return value.
    
    Examples:
    --------
    >>> Basic example of usage
    
    Notes:
    -----
    Mathematical description and references.
    """
    # Implementation steps:
    
    # 1. Input validation
    X, y = validate_inputs(X, y)
    
    # 2. Initialize parameters
    params = initialize_parameters(X, y, **hyperparameters)
    
    # 3. Core algorithm implementation
    results = _core_algorithm(X, y, params)
    
    # 4. Post-process results
    final_results = post_process(results)
    
    # 5. Return formatted outputs
    return format_outputs(final_results)
```

## 2. Implementation Best Practices

- Separate core algorithm logic from input validation and output formatting
- Use vectorized operations (NumPy/SciPy) wherever possible
- Include numerical stability safeguards
- Document mathematical equations using LaTeX
- Add time/space complexity analysis in comments
- Include references to original papers

## 3. Performance Considerations

- Mark code sections where performance is critical with:
  ```python
  # PERFORMANCE-CRITICAL-SECTION: [Description]
  ```

- Flag potential GPU acceleration opportunities:
  ```python
  # GPU-CANDIDATE: [Reason this could benefit from GPU]
  ```

The AI assistant will recognize these patterns and provide optimized implementations.
```

### 5. AI-Assisted Benchmark & Performance Testing

```python
# scripts/benchmark_models.py

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import importlib
import openai
import os
import json

def run_benchmark(module_path, function_name, dataset_sizes, iterations=3):
    """Run performance benchmark on a function with increasing dataset sizes."""
    # Dynamically import the module
    module_name = module_path.stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get the function
    function = getattr(module, function_name)
    
    results = []
    
    for size in dataset_sizes:
        # Generate synthetic dataset
        X = np.random.rand(size, 20)  # 20 features
        y = np.random.rand(size)
        
        # Run multiple iterations and take average
        times = []
        for _ in range(iterations):
            start_time = time.time()
            function(X, y)
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        results.append({
            'size': size,
            'time': avg_time,
            'samples_per_sec': size / avg_time
        })
    
    return pd.DataFrame(results)

def analyze_performance(benchmark_results, function_name, module_path):
    """Use AI to analyze performance results and suggest improvements."""
    # Read the function code
    with open(module_path, 'r') as f:
        code = f.read()
    
    # Convert benchmark results to string representation
    results_str = benchmark_results.to_string()
    
    # Create prompt for AI analysis
    prompt = f"""
    You are a machine learning performance optimization expert. Analyze the performance 
    benchmark results below for the function '{function_name}' and suggest specific 
    improvements. The benchmark was run with increasing dataset sizes.
    
    Benchmark results:
    {results_str}
    
    Function code:
    ```python
    {code}
    ```
    
    Please provide:
    1. Analysis of the scaling behavior (is it linear, quadratic, etc.?)
    2. Identification of performance bottlenecks in the code
    3. Specific optimization suggestions with code examples
    4. Recommendations for algorithm improvements or alternatives
    5. Potential parallelization or vectorization opportunities
    
    Focus on ML-specific optimizations that would improve this algorithm's performance
    without changing its mathematical properties or output quality.
    """
    
    # Call OpenAI API
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    response = openai.Completion.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=1500
    )
    
    return response.choices[0].text.strip()

def plot_results(benchmark_results, function_name, output_dir):
    """Create performance visualization plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot execution time vs dataset size
    ax1.plot(benchmark_results['size'], benchmark_results['time'], marker='o')
    ax1.set_xlabel('Dataset Size (samples)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title(f'{function_name} Execution Time')
    ax1.grid(True)
    
    # Plot throughput (samples/second)
    ax2.plot(benchmark_results['size'], benchmark_results['samples_per_sec'], marker='o')
    ax2.set_xlabel('Dataset Size (samples)')
    ax2.set_ylabel('Throughput (samples/second)')
    ax2.set_title(f'{function_name} Throughput')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / f"{function_name}_benchmark.png"
    plt.savefig(output_path)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Run performance benchmarks on ML algorithms')
    parser.add_argument('--module', required=True, help='Path to module containing function')
    parser.add_argument('--function', required=True, help='Function name to benchmark')
    parser.add_argument('--output', required=True, help='Output directory for results')
    args = parser.parse_args()
    
    module_path = Path(args.module)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define dataset sizes to test
    dataset_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
    
    # Run benchmark
    print(f"Benchmarking {args.function} in {module_path}...")
    results = run_benchmark(module_path, args.function, dataset_sizes)
    
    # Save results as CSV
    results_path = output_dir / f"{args.function}_benchmark.csv"
    results.to_csv(results_path, index=False)
    print(f"Benchmark results saved to {results_path}")
    
    # Generate plots
    plot_path = plot_results(results, args.function, output_dir)
    print(f"Plots saved to {plot_path}")
    
    # Analyze performance with AI
    print("Analyzing performance with AI...")
    analysis = analyze_performance(results, args.function, module_path)
    
    # Save analysis
    analysis_path = output_dir / f"{args.function}_analysis.md"
    with open(analysis_path, 'w') as f:
        f.write(f"# Performance Analysis for {args.function}\n\n")
        f.write(analysis)
    
    print(f"AI analysis saved to {analysis_path}")

if __name__ == "__main__":
    main()
```

## Example ML Framework Feature with AI-Assisted Development

Here's an example of how a complete ML framework feature would be developed with AI assistance:

### 1. Core Algorithm Implementation

```python
# src/models/gradient_boosting.py

import numpy as np
from ..metrics import mean_squared_error
from ..tree import DecisionTreeRegressor

def gradient_boosting_regressor(
    X, 
    y, 
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3, 
    min_samples_split=2,
    subsample=1.0,
    loss='squared_error',
    random_state=None
):
    """
    AI-DOC-TEMPLATE: Algorithm Implementation
    
    Gradient Boosting for regression.
    
    Builds an additive model in a forward stage-wise fashion;
    it allows for the optimization of arbitrary differentiable loss functions.
    
    Parameters:
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
        
    y : array-like of shape (n_samples,)
        Target values.
        
    n_estimators : int, default=100
        The number of boosting stages to perform.
        
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        
    max_depth : int, default=3
        Maximum depth of the individual regression estimators.
        
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
        
    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0, this results in Stochastic Gradient
        Boosting.
        
    loss : {'squared_error', 'absolute_error'}, default='squared_error'
        Loss function to be optimized.
        
    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to each Tree estimator at each
        boosting iteration.
    
    Returns:
    -------
    models : list
        List of fitted DecisionTreeRegressor instances.
        
    initial_prediction : float
        The initial prediction (usually mean of target).
    
    Examples:
    --------
    >>> import numpy as np
    >>> from ml_framework.models import gradient_boosting_regressor
    >>> from ml_framework.model_selection import train_test_split
    >>> X = np.random.rand(100, 4)
    >>> y = np.random.rand(100)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> models, initial_pred = gradient_boosting_regressor(X_train, y_train, n_estimators=10)
    
    Notes:
    -----
    This implementation follows the algorithm described in Friedman (2001).
    
    References:
    ----------
    J.H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine",
    The Annals of Statistics, Vol. 29, No. 5, 2001.
    """
    # Convert inputs to numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)