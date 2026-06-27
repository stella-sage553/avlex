# Security Policy

## Supported versions

avlex is pre-1.0; security fixes land on the latest released version.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Instead, use GitHub's
private vulnerability reporting ("Report a vulnerability" under the Security tab)
so it can be triaged privately.

Include a description, affected version, and a minimal reproduction if possible.
You can expect an initial response within a few days.

## Scope notes

avlex executes no untrusted input by design: it processes numeric arrays and
assembles text prompts. The optional `OpenAIClient` sends prompt text to a
third-party API — review your data-handling requirements before enabling it.
