# Contributing to Plumb

Thanks for considering a contribution. Plumb aims to stay small, dependency-free
at its core, and easy to drop into a pipeline.

## Setup

```bash
git clone https://github.com/sarcascoder/plumb
cd plumb
pip install -e ".[dev]"
pytest -q
```

## Principles

1. **Core stays stdlib-only.** Anything in `plumb/` (except optional extractor
   adapters) must run without third-party dependencies. Put heavy deps behind a
   lazy import inside an adapter, like `DoclingExtractor` does.
2. **Every behaviour has a test.** New metrics, confidence signals, or adapters
   ship with unit tests that run offline (no model calls).
3. **Be honest in docs.** Plumb is a focused library, not an IDP platform. Don't
   oversell it; point people to the right tool when it isn't us.

## Good first issues

- New `FieldType` comparison strategies (e.g. phone numbers, IBANs, enums).
- Extractor adapters (Marker, LlamaParse, Mistral OCR output).
- A document-image viewer for the review UI.
- Calibration improvements (embedding-similarity confidence signal).

## Pull requests

Keep PRs focused. Run `pytest -q` before opening. Describe what you changed and
why in plain language.
