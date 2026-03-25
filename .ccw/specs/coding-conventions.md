---
title: Coding Conventions
readMode: optional
priority: medium
category: general
scope: project
dimension: specs
keywords:
  - convention
  - typing
  - style
  - lint
  - naming
  - file
  - doc
---

# Coding Conventions

- [rule:typing] Use type hints for all function signatures and class attributes
- [rule:style] Prefer pure functions, list comprehensions, avoid mutable state
- [rule:style] Prefer early returns / guard clauses over deep nesting
- [rule:lint] Strict PEP 8 compliance with max line length 88 (Black formatter)
- [rule:naming] Variables and functions use snake_case (e.g., get_user_name)
- [rule:naming] Classes and types use PascalCase (e.g., UserService)
- [rule:naming] Constants use UPPER_SNAKE_CASE (e.g., MAX_RETRIES)
- [rule:file] Tests in a dedicated tests/ directory
- [rule:doc] All public functions and classes must have docstrings
- [rule:doc] Code should be self-explanatory; comments only for non-obvious logic
