---
title: Architecture Constraints
readMode: optional
priority: medium
category: planning
scope: project
dimension: specs
keywords:
  - constraint
  - arch
  - build
  - perf
  - security
---

# Architecture Constraints

- [rule:arch] Strict layer separation: UI → Service → Data (no skipping layers)
- [rule:arch] Modules must not have circular dependencies
- [rule:arch] Use dependency injection for testability, no hardcoded dependencies
- [rule:arch] Service/business logic must be stateless (state in DB/cache only)
- [rule:build] Adding new dependencies requires explicit justification and review
- [rule:perf] Large modules/routes must use lazy loading / code splitting
- [rule:security] All user input must be validated and sanitized before use
- [rule:security] No secrets in code — use env vars
