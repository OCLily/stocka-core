# Security

## Reporting a problem

Please report a security problem through GitHub:

- For anything that is safe to discuss in public, open an issue in this repository.
- For a vulnerability that should not be public before a fix, use GitHub's private vulnerability reporting ("Report a vulnerability" on the repository's Security tab).

Please include the affected file, the steps to reproduce and the Python version. There is no e-mail contact for reports.

## Scope

The code makes no network calls and needs no secrets: no API keys, tokens or credentials. The `stocka demo` command reads only the files you pass with `--config` or `--data-dir` and writes only the report file you name with `--out`. Without `--data-dir` it runs on generated synthetic data.

This is a research excerpt, not a product.
