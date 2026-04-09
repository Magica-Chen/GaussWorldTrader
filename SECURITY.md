# Security Policy

## Supported Versions

Only the latest version on the `main` branch receives security fixes.

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

If you discover a security issue in Gauss World Trader, please report it responsibly by emailing **zexun.chen@gauss.world** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Any suggested fix or mitigation (optional)

You can expect an acknowledgement within **72 hours** and a fix or status update within **14 days**.

## Scope

This policy covers security vulnerabilities in the Gauss World Trader source code. Issues relating to third-party dependencies should be reported to the relevant upstream project.

## Important Note on API Keys

Never commit API keys, secrets, or credentials to the repository. Use the provided `.env.example` template and keep your `.env` file local (it is already listed in `.gitignore`).
