# Plumb

**Self-hosted accuracy evals + a correction loop for document extraction. Bring your own model.**

You moved document extraction in-house — onto a local VLM, Docling, Marker, or your own pipeline on vLLM/Ollama — because it's ~167× cheaper per page than cloud APIs and your documents can't leave the building. But the moment you self-host, you lose the one thing the managed APIs quietly gave you: **confidence that the output is actually correct.**

Plumb is a small, code-first layer that **measures** how good your extraction is, field by field, and gives you a **correction loop** that turns human fixes into ground truth and few-shot examples. Everything runs locally. Nothing leaves your machine. The core has **zero dependencies**.

> Plumb is *complementary* to extractors, not a competitor. Point it at whatever you already use.

### Where Plumb fits (honest positioning)

If you want a full intelligent-document-processing **platform** — workflow builder, hosted review, connectors — use [Unstract](https://unstract.com) (open source) or commercial tools like Extend / Rossum / Nanonets. They are mature and excellent.

Plumb is deliberately the opposite: a **tiny library + CLI** you `import`, not a platform you adopt. Reach for it when you want document-aware accuracy metrics (table row/cell F1, currency/date/fuzzy normalization) and a confidence + review loop **as code, in your repo, in CI** — without standing up a whole platform. Generic LLM-eval libraries (`deepeval`, `pydantic-evals`) don't specialize in document field extraction; the IDP platforms aren't a `pip install` you wire into a pytest. Plumb sits in that gap.

---

## Why this exists

- Extractors (Docling, Marker, Unstract, LlamaParse) are excellent and commoditized. The hard part is no longer getting JSON out — it's **knowing the JSON is right.**
- Prompt-eval tools (promptfoo, deepeval) are built for chat/RAG, not document field extraction (currency, dates, multi-row line-item tables), and have no built-in human review.
- Model self-reported confidence is unreliable — LLMs are overconfident regardless of prompting. Plumb derives confidence from signals that actually correlate with correctness.

## What it does

1. **Field-level metrics** — type-aware comparison so formatting noise doesn't count as error:
   - `exact` (ids/codes), `number` (currency + tolerance), `date` (format-invariant), `string` (fuzzy + threshold), `table` (row alignment → precision/recall/F1 on line items).
2. **Calibrated confidence** — self-consistency across repeated runs flags exactly which fields a human should review; a reliability table + **safe auto-accept threshold** tells you where you can stop reviewing.
3. **Correction loop** — corrections are stored locally (SQLite) and become both ground truth for future evals and few-shot examples for the extractor.
4. **CI gate** — `plumb eval --min-accuracy 0.95` exits non-zero, so an extraction regression fails your build.

## Quickstart

```bash
./run.sh          # macOS/Linux: demo + review UI   (run.bat on Windows)
# or, manually:
python -m examples.invoices.run_demo
```

```python
from plumb import FieldSpec, FieldType, evaluate
from plumb.report import text_report

schema = [
    FieldSpec("invoice_number", FieldType.EXACT),
    FieldSpec("vendor",         FieldType.STRING, threshold=0.85),
    FieldSpec("invoice_date",   FieldType.DATE),
    FieldSpec("total",          FieldType.NUMBER, abs_tol=0.01),
    FieldSpec("line_items",     FieldType.TABLE, columns=(
        FieldSpec("desc",   FieldType.STRING),
        FieldSpec("amount", FieldType.NUMBER),
    )),
]

predictions  = {"inv_001": {"invoice_number": "A-1001", "vendor": "ACME, Inc",
                            "invoice_date": "01/15/2026", "total": "$1,200.00", ...}}
ground_truth = {"inv_001": {"invoice_number": "A-1001", "vendor": "Acme Inc.",
                            "invoice_date": "2026-01-15", "total": 1200.00, ...}}

print(text_report(evaluate(schema, predictions, ground_truth)))
```

### Bring your own model

```python
from plumb.extractors import OpenAICompatExtractor

# works with Ollama, vLLM, llama.cpp server, or your RunPod endpoint
extractor = OpenAICompatExtractor(base_url="http://localhost:11434/v1",
                                  model="qwen2.5-vl")
prediction = extractor.extract(document_text, schema)
```

### Use it in your pipeline (the `Pipeline` facade)

One object wires extractor + schema + a local store together. This is the
intended integration point:

```python
from plumb import Pipeline, FieldSpec, FieldType
from plumb.extractors import OpenAICompatExtractor

pipe = Pipeline(
    schema=[FieldSpec("invoice_number", FieldType.EXACT),
            FieldSpec("total", FieldType.NUMBER, abs_tol=0.01)],
    extractor=OpenAICompatExtractor(model="qwen2.5-vl", temperature=0.3),
    store_path="plumb.db",
    consistency_runs=3,            # >1 → self-consistency confidence per field
)

pipe.extract("inv_001", document_text)     # run model, store prediction(s)
for item in pipe.review_queue():           # worst-first: what a human should check
    print(item.doc_id, [f.name for f in item.fields if f.reason])

pipe.record_correction("inv_001", {"total": 1200.00})   # → ground truth + few-shot
report = pipe.evaluate()                   # field-level accuracy vs your corrections
block  = pipe.fewshot_block({"inv_001": document_text})  # prime the next extraction
```

Drop `report.document_accuracy` into an assert and you have a regression gate in
your own test suite.

### CLI

```bash
# score predictions against ground truth (exit 1 if below target → CI gate)
plumb eval --pred preds.json --truth truth.json --schema schema.json --min-accuracy 0.95

# open the review UI on your own data (omit flags to use the bundled demo)
plumb review --schema schema.json --pred preds.json --samples samples.json
```

### Continuous integration

`plumb eval --min-accuracy` returns a non-zero exit code when accuracy drops,
so a regression fails the build. A ready-to-edit GitHub Actions workflow lives at
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) — point it at your own
`schema.json` / `predictions.json` / `ground_truth.json` and set your threshold.

## Architecture

```
your extractor ──► predictions ─┐
                                 ├─► plumb.metrics  ─► per-field accuracy, weakest fields
ground truth (humans) ──────────┘     plumb.confidence ─► review queue, auto-accept threshold
                                       plumb.store    ─► local SQLite, corrections feed back
```

- **Core: pure Python stdlib.** No GPU, no always-on service, no data egress.
- **Adapters** wrap any extractor. Cloud or local — Plumb doesn't care.
- **Review UI**: zero-dependency, built on the stdlib `http.server`. No FastAPI required.

### Review UI

```bash
plumb review              # opens the local review queue at http://127.0.0.1:8000
```

Worst-first queue, each field annotated with *why* it's flagged (disagrees with
ground truth, or low self-consistency confidence). Edit, click **Save as ground
truth** — the correction is written locally and feeds future evals.

### Model leaderboard (on your documents)

```python
from plumb import compare_models, leaderboard_text
lb = compare_models(schema, ground_truth, {"qwen2.5-vl": preds_a, "granite-docling": preds_b})
print(leaderboard_text(lb))   # ranks models AND picks the best model per field
```

## Roadmap

- [x] Field-level metrics engine (exact / number / date / string / table)
- [x] Self-consistency confidence + reliability table + safe auto-accept threshold
- [x] Local SQLite store with correction write-back
- [x] OpenAI-compatible extractor adapter (Ollama / vLLM / RunPod)
- [x] CLI with CI gate
- [x] Web review UI (document view + field correction) — stdlib, zero deps
- [x] Docling adapter + generic mapping adapter (evaluate any extractor's output)
- [x] Model/prompt leaderboard on *your* documents (best model per field)
- [x] Few-shot example export from corrections (`plumb.fewshot`) — feeds verified
      fixes back into the extractor prompt; accuracy compounds as you review
- [ ] Side-by-side document image viewer in the review UI
- [ ] Marker adapter + a published PyPI release

## License

Apache-2.0. Core is and stays open source.
