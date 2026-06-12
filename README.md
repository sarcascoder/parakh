# parakh

**Self-hosted, bring-your-own-model OCR & extraction evaluation — with human-in-the-loop correction and CI gates.**

> *parakh (परख)* — Hindi for *test, scrutinise, judge.* What your evals should actually do.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/parakh.svg)](https://pypi.org/project/parakh/)

---

## The problem

You shipped OCR / VLM / document extraction to production. Now what?

- Accuracy numbers from the model vendor are on **their** benchmark, not your corpus.
- Your "evaluation" is three engineers hand-checking 20 documents on Friday.
- When the model drifts or the vendor pushes an update, you find out from a customer.
- "Field-level accuracy" means different things to different teams; nobody agrees on a number.

parakh fixes this. **It is the evaluation framework I built because I needed one and the alternatives were either toy benchmarks or $40K enterprise platforms.**

---

## What parakh does

- **Field-level metrics** — character-level accuracy, exact match, fuzzy match, IoU on bounding boxes, all per-field. Not a single hallucinated F1.
- **Confidence calibration** — flag low-confidence predictions for human review; track whether the model's confidence actually correlates with correctness over time.
- **Human correction loop** — annotate, correct, save back to a golden set. The golden set is the only asset that compounds.
- **CI gate** — a single command. Fails your PR if accuracy regresses past a threshold you set.
- **BYO-model** — all major OCR engines and the leading open-source vision-language models, plus your own. Adapter pattern, ~30 lines per new model.
- **Self-hosted** — your data never leaves your infra.

---

## Quick start

```bash
pip install parakh
parakh init my-eval/
# put your documents in my-eval/inputs/
# put your golden outputs in my-eval/golden/
parakh eval --model your-vlm --config my-eval/config.yaml
parakh dashboard  # local web UI for review + correction
```

CI gate:

```yaml
# .github/workflows/eval.yml
- uses: sarcascoder/parakh-action@v1
  with:
    model: your-vlm
    threshold: 0.92  # fails PR if accuracy drops below
```

---

## What's coming (paid)

The OSS version is intentionally complete for single-team self-hosted use. For everything beyond that:

**parakh Cloud** *(early access, [join waitlist →](https://parakh.cloud))*

- Hosted dashboard, history, dataset versioning
- Multi-team RBAC + audit log
- Side-by-side model comparison across versions
- Slack/email alerts on regression
- SOC 2-friendly architecture, EU + US data residency
- Pricing: $99 / $499 / $1,499 per month

If you're hand-rolling evals or paying enterprise prices for less, get on the list.

---

## What this is not

- **Not a labelling tool.** Use Label Studio for that. parakh consumes your golden set; it doesn't help you create it from zero.
- **Not a training framework.** parakh evaluates. Train wherever you train.
- **Not opinionated about your stack.** Adapters for everything I use in production, easy to add more.

---

## Used in production by

Hashteelab (manufacturing, automotive, cement, legal clients) — and counting.

## Works hand-in-hand with [OpenExtract](https://github.com/sarcascoder/openextract)

If you're using [OpenExtract](https://github.com/sarcascoder/openextract) (or any self-hosted Textract / Azure DocInt / Google Doc AI alternative), parakh is the eval framework that proves it works on *your* corpus. Same author. Same family.

## License

Apache-2.0 for the OSS. parakh Cloud is a separate hosted commercial service.

📧 **tanupam760@gmail.com** · [GitHub](https://github.com/sarcascoder) · [parakh.cloud](https://parakh.cloud)
