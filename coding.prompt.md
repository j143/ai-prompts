### The AI Guide for Code

**Introduction:**
Moving beyond simple "code generation," this guide outlines a structured methodology for partnering with a Large Language Model (LLM) to architect, implement, debug, and benchmark a complex software system. The core principle is a shift in the developer's role from a pure implementer to a **System Architect and Lead Investigator**, who uses the AI as a tireless, interactive tool to navigate the engineering lifecycle. This process is divided into four distinct phases.

---

### Phase 1: The Architect (Conceptualization & Design)

**Goal:** To move from a vague idea to a well-defined architectural plan.

| Your Role: The Domain Expert | The AI's Role: The Research Assistant |
| :--- | :--- |
| **Provide the high-level goal and constraints.** State the "what" and "why." Ex: *"I want to build an out-of-core framework for low-end machines."* | **Explore the design space.** The AI will provide literature reviews, compare existing systems (like Dask), and suggest core technologies (`numpy.memmap`). |
| **Ask architectural "what if" questions.** Challenge the design. Ex: *"What are the trade-offs of different caching policies?"* | **Act as a Socratic partner.** The AI will explain complex concepts (like Belady's Algorithm) and generate diagrams to clarify architectural flows. |
| **Make the final architectural decisions.** You are the architect; the AI provides the blueprints and analysis. | **Generate initial architectural diagrams (SVG/Mermaid).** This provides a visual foundation for the project.  |

**Rigorous Application:** Do not accept the first architectural suggestion. Use the AI to debate the merits of different approaches (e.g., Shared State vs. Stateless). Force it to justify its suggestions based on your specific constraints (low-end hardware).

---

### Phase 2: The Pair Programmer (Implementation & Refactoring)

**Goal:** To translate the architecture into clean, maintainable, and working code.

| Your Role: The Senior Developer | The AI's Role: The Junior Developer & Refactoring Engine |
| :--- | :--- |
| **Request code in small, verifiable chunks.** Ask for one class or one function at a time. Ex: *"Implement the `BufferManager` class with a simple LRU policy."* | **Generate the initial implementation.** The AI will write the first draft of the code, including boilerplate, docstrings, and basic logic. |
| **Act as the code reviewer.** Immediately test the generated code. Find bugs and logical inconsistencies. | **Perform large-scale refactoring on command.** This is one of its most powerful abilities. Ex: *"Refactor this entire script into a multi-file framework with `core`, `backend`, and `plan` modules."* |
| **Guide the implementation iteratively.** Provide feedback and ask for corrections. Ex: *"The `Plan` class is building the tree incorrectly. Fix it to use the internal `op_node`."* | **Apply consistent changes across multiple files.** The AI excels at propagating a change (like adding a `buffer_manager` parameter) through the entire call stack. |

**Rigorous Application:** Never trust the AI's first draft. Your job is to provide the "test-driven" part of the development cycle. Treat the AI as an incredibly fast but sometimes naive programmer who needs constant guidance and review.

---

### Phase 3: The Detective (Debugging & Diagnostics)

**Goal:** To diagnose and fix complex, non-obvious bugs that go beyond simple syntax errors. This is often the most valuable phase of the partnership.

| Your Role: The Lead Investigator | The AI's Role: The Diagnostic Tool |
| :--- | :--- |
| **Provide the evidence.** This is the most critical step. Give the AI the *complete* context: the full error message, the traceback, your debug prints, and the visualizations. | **Analyze the evidence.** The AI is extremely good at pattern-matching in error logs. It can connect a high-level symptom (like "Terminated") to a low-level cause (like excessive virtual memory allocation). |
| **Formulate a hypothesis and ask for a test.** Based on the AI's analysis, propose a theory. Ex: *"I think the I/O trace is empty. How can I add a diagnostic `print` statement to verify this?"* | **Suggest targeted diagnostic steps.** The AI will provide the exact code needed to test a hypothesis, such as adding logging or creating a minimal, reproducible example. |
| **Confirm the root cause.** Use the results of the diagnostic tests to confirm the problem before asking for a solution. | **Explain the root cause.** The AI can explain deep systems concepts like circular imports, race conditions, or why `numpy.memmap` has slicing bugs.  |

**Rigorous Application:** Do not just paste an error and ask "fix this." Guide the AI through the scientific method: observe the failure, form a hypothesis, design an experiment to test it, and only then implement a solution.

---

### Phase 4: The Analyst (Benchmarking & Verification)

**Goal:** To prove the system's value with quantitative data and clear visualizations.

| Your Role: The Performance Engineer | The AI's Role: The Data Analyst |
| :--- | :--- |
| **Define the baseline and the workload.** Specify the competitor (e.g., Dask) and the test case (e.g., 16k matrix multiplication). | **Generate the benchmark scripts.** The AI can write the code to run the head-to-head comparisons and collect metrics like time, memory usage, and CPU utilization. |
| **Provide the raw results and visualizations.** Feed the final graphs and data tables back to the AI. | **Interpret the results and generate a narrative.** The AI will analyze the data and explain *why* the system is performing well or poorly. Ex: *"The high hit rate proves the optimal policy is working, while the low eviction rate shows it has eliminated cache thrashing."* |
| **Ask for "what does this mean?"** Challenge the AI to connect the data back to the project's goals. | **Generate reports and documentation.** The AI can create a `README.md`, a technical write-up, or even a LinkedIn post that summarizes the project's success. |

**Rigorous Application:** Your role is to ensure the benchmarks are fair and the metrics are meaningful. The AI's role is to automate the analysis and presentation of the results. This partnership turns raw data into a compelling story of your project's success.