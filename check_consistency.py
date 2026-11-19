"""Check saved artifacts and the application's example against notebook metadata.

Run with Python 3.12 after installing requirements.txt:
    python check_consistency.py

This checks artifact loading and inference consistency. It does not establish
that the notebook's evaluation split is free of candidate overlap.
"""
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
import spacy
from packaging.specifiers import SpecifierSet

import nlp_pipeline as nlpp

BASE = Path(__file__).resolve().parent


def check(ok: bool, message: str) -> bool:
    print(("OK   " if ok else "FAIL ") + message)
    return bool(ok)


def main() -> int:
    metadata = json.loads((BASE / "model_metadata.json").read_text(encoding="utf-8"))
    model_meta = json.loads((BASE / "nlp_model" / "meta.json").read_text(encoding="utf-8"))
    versions = metadata["versions"]
    checks = [
        check(sklearn.__version__ == versions["sklearn"],
              f"scikit-learn {sklearn.__version__}; artifact: {versions['sklearn']}"),
        check(np.__version__ == versions["numpy"],
              f"NumPy {np.__version__}; artifact: {versions['numpy']}"),
        check(spacy.__version__ in SpecifierSet(model_meta["spacy_version"]),
              f"spaCy {spacy.__version__}; model requires {model_meta['spacy_version']}"),
        check("fallback" not in nlpp.MODEL_INFO["ner_model"]
              and nlpp.MODEL_INFO["tfidf"].startswith("tfidf.pkl"),
              f"Saved artifacts loaded: {nlpp.MODEL_INFO}"),
        check(len(nlpp._tfidf.vocabulary_) == metadata["tfidf"]["n_terms"],
              "TF-IDF vocabulary size matches metadata"),
    ]
    for name, expected in metadata["tfidf"]["params"].items():
        actual = nlpp._tfidf.get_params()[name]
        if isinstance(actual, tuple):
            actual = list(actual)
        checks.append(check(actual == expected, f"TF-IDF {name} = {actual}"))

    example = metadata["example"]
    result = nlpp.analyze_texts(example["resume_text"], example["job_text"])
    for name in ("tfidf_sim", "skill_overlap", "final_score"):
        checks.append(check(abs(result[name] - example[name]) < 1e-6,
                            f"Notebook example {name}: {result[name]} (expected {example[name]})"))

    notebook_model, source = nlpp.load_spacy_model(extend=False)
    checks.append(check(len(notebook_model.get_pipe("entity_ruler").patterns)
                        == metadata["ner"]["n_patterns"], "Saved NER pattern count matches metadata"))
    for text in (example["resume_text"], example["job_text"]):
        checks.append(check(nlpp.extract_skills(text) == nlpp.extract_skills(text, notebook_model),
                            "Application and saved NER agree on example skills"))
    print("\nAll consistency checks passed." if all(checks) else "\nConsistency check failed.")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
