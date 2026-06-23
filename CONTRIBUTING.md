# Contributing to local-semantic-cache

First off, thank you for considering contributing to `local-semantic-cache`! 

We welcome contributions of all types:
*   Adding support for pre-packaged adapter wrappers (like LangChain or Vercel AI SDK adapters).
*   Improving performance of the local cosine similarity math logic.
*   Fixing bugs in text normalization, tokenization, or file persistence.
*   Writing documentation, guides, or visual test cases.

---

## Local Setup

To set up the project locally:

1.  Fork and clone the repository:
    ```bash
    git clone https://github.com/Ezeko/semanticcache.git
    cd semanticcache
    ```
2.  Install development dependencies:
    ```bash
    npm install
    ```
3.  Create a branch for your feature:
    ```bash
    git checkout -b feature/my-cool-improvement
    ```

---

## Compiling & Testing

Since this project is written in TypeScript, you need to compile it before running tests or checking code validity:

*   **Build the project**:
    ```bash
    npm run build
    ```
    This compiles the TS source files into the `./dist/` directory.
*   **Run automated tests**:
    ```bash
    npm test
    ```
    This automatically builds the project and executes the native Node.js test runner against compiled files inside `dist/tests/`.
*   **Run visual demo script**:
    ```bash
    npm run build && node dist/scratch_run.js
    ```

---

## Coding Rules & Guidelines

*   **Strict Type-Safety**: Ensure all interfaces, classes, and helper parameters are fully typed. Avoid using `any` type overrides.
*   **Zero Dependencies**: We aim to keep `local-semantic-cache` light, secure, and fast. Please do not add external dependency modules to `package.json`.
*   **Vector Normalization**: If you modify the built-in vectorizer, ensure that output vectors are **L2 unit normalized** (sum of squares = 1.0). This is critical because our similarity search is optimized to run as a fast dot product (which requires unit-length vectors).

Thank you for contributing!
