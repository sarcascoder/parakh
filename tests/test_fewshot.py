from parakh.fewshot import build_examples, render_block, examples_covering_fields


GT = {
    "d1": {"invoice_number": "A-1", "vendor": "Acme", "total": 100},
    "d2": {"invoice_number": "B-2", "vendor": "Globex", "total": 50},
    "d3": {"invoice_number": "C-3", "vendor": "", "total": 75},
}
DOCS = {
    "d1": "Invoice A-1 from Acme, total 100",
    "d2": "Invoice B-2 from Globex, total 50",
    # d3 has no source text on file
}


def test_build_examples_only_pairs_with_known_text():
    ex = build_examples(GT, DOCS)
    ids = {e.output["invoice_number"] for e in ex}
    assert ids == {"A-1", "B-2"}          # d3 skipped (no document text)


def test_prefer_docs_ordering_and_limit():
    ex = build_examples(GT, DOCS, limit=1, prefer_docs=["d2"])
    assert len(ex) == 1
    assert ex[0].output["invoice_number"] == "B-2"


def test_render_block_is_prompt_injectable():
    block = render_block(build_examples(GT, DOCS))
    assert "verified examples" in block
    assert "A-1" in block and "Acme" in block


def test_examples_covering_fields_prioritises_weak_fields():
    # weak field = vendor; d3 has empty vendor so it should rank lowest
    ex = examples_covering_fields(GT, {**DOCS, "d3": "Invoice C-3 total 75"},
                                  weak_fields=["vendor"], limit=2)
    vendors = [e.output["vendor"] for e in ex]
    assert "" not in vendors          # the empty-vendor doc is not chosen first
