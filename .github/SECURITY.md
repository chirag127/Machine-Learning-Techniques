# Security Policy for ML-Concept-Visualizer-Static-Site-Template

As the Apex Technical Authority, security is not an afterthought; it is the foundational layer upon which all robust systems are built. This policy outlines how security vulnerabilities in `ML-Concept-Visualizer-Static-Site-Template` are reported, assessed, and remediated.

## 1. Supported Versions

This project prioritizes the use of modern, actively maintained libraries and frameworks (Vite 7+, TypeScript 5.x+, standard web APIs). 

| Version | Support Status |
| :--- | :--- |
| Latest Stable Release | Fully Supported |
| Previous Minor Release | Supported for Critical CVEs only |
| Versions older than N-1 | Unsupported. Users should upgrade immediately. |

## 2. Reporting a Vulnerability

We welcome responsible disclosure of security vulnerabilities. By following these guidelines, you help us maintain the integrity and security posture of this template.

### A. Preferred Reporting Channel

Please report all security concerns through the following channels in order of preference:

1.  **GitHub Security Advisory:** Create a private report via the GitHub Security tab in this repository: `https://github.com/chirag127/ML-Concept-Visualizer-Static-Site-Template/security/advisories/new`
2.  **Direct Email:** If GitHub reporting is not possible, email the maintainer directly: `security+apex@example.com` (Replace with actual security contact if established).

### B. Required Information for Disclosure

To facilitate rapid triage and resolution, please include the following details in your report:

*   **Vulnerability Type:** (e.g., XSS, CSRF, Dependency Confusion, Supply Chain Risk).
*   **Affected Component:** Specify the exact file, library, or feature.
*   **Proof of Concept (PoC):** Clear, concise steps to reproduce the vulnerability.
*   **Impact Assessment:** Describe the potential harm if exploited.
*   **Recommended Mitigation (If known):** Your suggested fix or patch.

## 3. VULNERABILITY RESPONSE TIMELINE (SLAs)

We adhere to an aggressive Service Level Agreement (SLA) for security incidents, reflecting the philosophy of "Zero-Defect, High-Velocity."

| Severity Level | Initial Response SLA | Remediation Goal SLA |
| :--- | :--- | :--- |
| **Critical (P0)** | 4 Hours | 48 Hours |
| **High (P1)** | 12 Hours | 5 Business Days |
| **Medium (P2)** | 24 Hours | 10 Business Days |
| **Low (P3)** | 3 Business Days | Best Effort / Next Minor Release |

*   **Critical Vulnerabilities** involving potential data exposure or execution on end-user machines will trigger an immediate internal high-priority incident response.

## 4. MITIGATION STRATEGY (Defense in Depth)

This template is built on a static architecture, reducing attack surface, but we employ layered security:

1.  **Dependency Scanning:** Automated scanning via GitHub Actions (`ci.yml`) to flag known CVEs in the dependency tree.
2.  **Linter Enforcement:** Strict use of TypeScript compiler options and Biome rules to eliminate common pitfalls like implicit `any` types and insecure JavaScript patterns.
3.  **Content Security Policy (CSP):** The default setup mandates a restrictive CSP to mitigate XSS risks inherently, even if a vector is discovered in a third-party library.
4.  **Principle of Least Privilege:** As a static site template, there is no server-side logic, eliminating entire classes of server-based attacks.

## 5. DEPENDENCY MANAGEMENT

Dependencies are managed via `package.json` and are subject to regular automated audits. Users upgrading this template are responsible for verifying dependency integrity. We strongly recommend using `npm audit fix` upon updating packages.

--- 

*Last Reviewed: December 2025. Maintained by chirag127.*