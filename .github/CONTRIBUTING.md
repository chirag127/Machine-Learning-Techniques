# 🚀 Contributing to ML-Concept-Visualizer-Static-Site-Template

Welcome, Architect. Your expertise is vital to advancing the state of ML documentation visualization. This project adheres to **Zero-Defect, High-Velocity, Future-Proof** engineering principles, aligning with the Apex Technical Authority standards.

## 1. Prerequisites & Environment Setup

Before contributing, ensure your local environment meets the project's core stack requirements (as defined in `AGENTS.md`):

1.  **Clone the Repository:**
    bash
    git clone https://github.com/chirag127/ML-Concept-Visualizer-Static-Site-Template.git
    cd ML-Concept-Visualizer-Static-Site-Template
    

2.  **Node.js/Vite Environment:** This project uses TypeScript and Vite for its static generation.
    bash
    # Install dependencies using the modern resolver
    npm install
    

3.  **Linting & Formatting Check:** Always verify compliance before committing.
    bash
    npm run lint
    npm run format
    

## 2. Development Workflow

We enforce **Feature-Sliced Design (FSD)** principles, even in a static context, to ensure scalability and maintainability of visualization components.

1.  **Branching Strategy:** All new work, features, or fixes must be implemented on a dedicated feature branch based off `main`.
    bash
    git checkout -b feature/short-descriptive-name
    

2.  **Atomic Commits:** Commits must be small, atomic, and descriptive. Follow the **Conventional Commits** specification (e.g., `feat: add new visualization module` or `fix: correct data mapping logic`).

3.  **Verification:** Before pushing, ensure all local tests pass (`npm run test:unit`) and the site builds successfully (`npm run build`).

## 3. Pull Request (PR) Submission Protocol

Use the provided PR Template (`.github/PULL_REQUEST_TEMPLATE.md`) as your guide. High-quality PRs are merged rapidly.

*   **Self-Review First:** Thoroughly review your own changes against the project's architectural principles (SOLID, DRY, YAGNI).
*   **CI Gate:** **Do not bypass CI checks.** The automated pipeline must pass green before a human review begins.
*   **Documentation Update:** If you introduce a new feature or fix a significant bug, you **MUST** update the relevant documentation sections (e.g., `README.md`, internal component documentation).

## 4. Reporting Issues and Bugs

If you encounter a defect or wish to propose an enhancement, please use the dedicated Issue Template (`.github/ISSUE_TEMPLATE/bug_report.md`).

*   **Clarity is Paramount:** Provide clear reproduction steps, expected results, and actual results. If reporting a visualization bug, provide clear screenshots or links to the artifact.

## 5. Architectural Integrity & Principles

Contributors are expected to uphold the following standards:

*   **SOLID Compliance:** Favor Single Responsibility Principle (SRP) in all visualization components.
*   **DRY Enforcement:** Abstract repeated styling or data transformation logic aggressively.
*   **YAGNI Adherence:** Do not over-engineer solutions for hypothetical future requirements. Build only what is necessary now.
*   **Security Posture:** As a static site, focus primarily on supply chain security (dependency updates) and front-end hardening.

Thank you for contributing to the `ML-Concept-Visualizer-Static-Site-Template`.