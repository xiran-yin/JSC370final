# Midterm Rubric Notes (For Final Project Writing)

These notes summarize the midterm feedback and should be treated as a checklist before submission.

## 1) Introduction

- Keep background and motivation concise and data-grounded.
- Write hypotheses in fully testable form:
  - define clear null and alternative
  - avoid hypotheses that do not map to a specific test/model

## 2) Methods

- Clearly state all data sources and provenance.
- Explicitly define data wrangling logic:
  - what "conditional parsing" means in concrete code terms
  - what exceptions were handled and how
  - how missing or inconsistent historical files were resolved
- Explicitly define engineered categories:
  - weather severity variables used
  - breakpoints/thresholds for each category
- When using decomposition or specialized methods, name:
  - the exact function/package
  - the model components and settings

## 3) Results

- Keep interpretation conservative and tied to estimated effects.
- Avoid over-claiming from visually weak differences.
- If possible, report estimated slopes/effect sizes rather than only visual claims.
- Discuss sensitivity to outliers for non-linear smoothers/LOWESS/GAM.
- If describing interactions, show explicit stratified lines/models.

## 4) Summary/Discussion

- Use plain, technical language; avoid gimmicky terms.
- Separate:
  - what is strongly supported,
  - what is suggestive,
  - what is uncertain.
- Include explicit limitations.
- Justify threshold choices (e.g., severe-day cutoff) and discuss alternatives.
- Compare classification vs regression framing when relevant.

## 5) Reproducibility and Style

- Keep project fully reproducible from repo files.
- Ensure all figures/tables are captioned and readable.
- Keep final report narrative coherent and methods transparent.

---

Working decision for the final version in this repo:

- NLP/reddit sentiment section is removed from final deliverables.
- Final scope is weather + TTC delay modeling with regression/classification and interactive visualizations.
