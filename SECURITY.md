# Security Policy

ASTScribe is designed as a static-analysis library. Analyzed source code must never be executed as part of analysis.

Any pathway that executes, imports, evaluates, or otherwise causes side effects from analyzed source should be treated as a security vulnerability.

Please report security-sensitive findings using GitHub's private vulnerability reporting / Security Advisory features when available. Do not include sensitive exploit details in a public issue before maintainers have had an opportunity to review them.
