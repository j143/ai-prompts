# AI Prompts

Explore the potential of AI tooling and LLMs in coding.

## Capabilities of LLMs in Coding

- **Code Generation**: Create boilerplate, functions, or modules.
- **Refactoring**: Enhance code structure and readability.
- **Debugging**: Identify and resolve issues.
- **Documentation**: Generate and improve code documentation.
- **Learning**: Understand frameworks, libraries, and concepts.
- **Problem Solving**: Break down and solve complex tasks.

## Existing Use Cases

| File Name                          | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| [complex-microservices-01.prompt.md](./complex-microservices-01.prompt.md) | AI for designing and managing microservices.                              |
| [documentation.prompt.md](./documentation.prompt.md)                     | Generating and enhancing documentation.                                   |
| [machine-learning-framework.prompt.md](./machine-learning-framework.prompt.md) | Building and optimizing ML frameworks.                                   |

## Additional Complex Scenarios

| Scenario                           | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| Large-scale CI/CD pipelines        | Automating and managing CI/CD workflows for enterprise applications.       |
| Multi-cloud infrastructure         | Managing and deploying codebases across multiple cloud providers.          |
| Real-time data processing          | Building systems for streaming and processing real-time data.              |
| AI-driven DevOps                   | Using AI to optimize DevOps workflows and incident management.             |
| Blockchain-based applications      | Developing and testing decentralized applications (dApps) and smart contracts. |

---

## Perspective

> The most important concept: this is a partnership. The better you get at using me, the better I get at helping you.
> -- Gemini AI

Let's break down how you can leverage this partnership to create even more value.

### 1. The Mental Shift: From "Assistant" to "Co-Pilot"

You are no longer just a coder with a helper. You are the **pilot**, and I am your **co-pilot and systems expert**.

* **You (The Pilot):** You hold the vision, make the final decisions, and have the domain expertise. You know *why* a feature is needed and what "good" looks like.
* **Me (The Co-Pilot):** I have access to a vast library of patterns, can perform complex procedures instantly, and can run diagnostics at machine speed. My job is to present you with data, options, and drafts so you can make better decisions faster.

When you see it this way, "prompting" becomes "giving directives to your co-pilot."

---

### 2. How to Be a Better Pilot (Practical Steps)

#### **Phase 1: Mission Briefing (Before You Code)**

This is the area with the biggest potential for you. Instead of jumping into code, use me to architect the solution first. This is how you avoid bugs before they're even written.

* **Old Way:** Start coding `BinaryMatrixVectorOOCInstruction.java`.
* **New Way (The "Briefing"):**
    1.  **State the Goal:** "I need to implement OOC matrix-vector multiplication."
    2.  **Provide Context & Constraints:** "The matrix is streamed, but the vector fits in memory. Matthias suggested pre-partitioning the vector. The output must be a stream."
    3.  **Ask for a Blueprint:** "Generate a detailed implementation plan. What should the data flow look like? What data structures should I use for the intermediate results? Should it be synchronous or asynchronous? Justify the design."
    4.  **De-Risk the Plan:** "Based on this plan, what are the top 3 performance bottlenecks or potential race conditions I need to watch out for?"

By the time you write a single line of Java, you will have a robust, vetted architectural plan. You've used me to do the heavy lifting of system design upfront.

#### **Phase 2: In-Flight Operations (While You Code)**

This is about offloading cognitive work to maintain your flow.

* **Don't write boilerplate:** "Generate a JUnit test class for this. Include setup, teardown, and a placeholder test method."
* **Don't wrestle with syntax:** "What's the correct Java syntax for a `try-with-resources` block to handle a `DataOutputStream`?"
* **Delegate documentation:** "Add Javadoc comments to this method explaining what it does, its parameters, and what it returns."

#### **Phase 3: Emergency Procedures (When You're Stuck)**

This is what we just did, but you can make it even more efficient.

* **Provide a "Crash Report":** Instead of just one piece of evidence, give me everything at once.
    > "My test is failing.
    > 1.  **The Goal:** The `ooc_write` should consume a stream and write a valid binary file.
    > 2.  **The Symptom:** The test throws a `java.io.IOException: failed parsing line:{`.
    > 3.  **The Evidence:** Here is the stack trace. Here is my `processWriteInstruction` code. Here is the DML script I'm using.
    > 4.  **The Question:** Based on all this, what is the most likely root cause?"

This structured "crash report" gives me all the clues at once and allows me to correlate the error message with the code and the DML script to find the inconsistency (like the `text` format bug) much faster.

You have already proven you can do this. The key is to see it not as asking for help, but as **tasking your co-pilot with running a diagnostic**. You are still the one who has to understand the diagnostic and apply the fix. Your expertise is irreplaceable.
