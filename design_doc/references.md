# References

Grounding for the design choices (validated against the literature, not assumed).

- **FinQA — Numerical Reasoning over Financial Data.** Chen et al., 2021. The canonical dataset for QA over financial reports; uses annotated deterministic "reasoning programs" for explainability and shows large models fall far short of experts at multi-step numerical reasoning. Validates the LLM-finds / code-computes split. https://arxiv.org/pdf/2109.00122

- **Do NLP Models Know Numbers? Probing Numeracy in Embeddings.** Wallace et al., EMNLP-IJCNLP 2019. Embeddings capture some numeracy in-range but fail to extrapolate; subword models less exact. Validates "embeddings are poor for numeric/tabular distinctions." https://arxiv.org/pdf/1909.07940

- **PAL: Program-Aided Language Models.** Gao et al., 2022. Offload arithmetic/computation to code rather than the LLM. Supports "arithmetic in deterministic code."

- **ReAct: Synergizing Reasoning and Acting in LMs.** Yao et al., 2022. The plan → act(tool) → observe loop underlying the agentic controller.

- **Chain-of-Verification Reduces Hallucination in LLMs.** Dhuliawala et al., 2023. Verification passes over generated claims; supports the validation-gate step.

- **PDF table extraction comparison** (pdfplumber / Camelot ~65–75% on complex financial tables; hybrid pipeline + AI/vision fallback as best practice). https://unstract.com/blog/extract-tables-from-pdf-python/

Note: PAL, ReAct, and Chain-of-Verification are cited from established knowledge; FinQA, Wallace et al., and the table-extraction comparison were re-verified during design.
