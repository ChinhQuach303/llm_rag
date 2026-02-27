---
description: Auto-update README.md and docs/results.md when a new performance record is achieved
---

# /update-docs - Auto Update Performance Records

$ARGUMENTS

---

## Purpose

This command is triggered to automatically update the performance documentation (`README.md` and `docs/results.md`) whenever a new model training, parameter tuning, or code enhancement yields better metrics than the current baseline.

---

## Pre-Requisites

Before running this workflow, ensure that:

1. A new training pipeline or evaluation has been completed successfully.
2. The newly acquired metrics (such as MAE, Pct_Short, Pct_Over) have been cross-verified to be objectively better than the currently documented numbers (e.g. lowering Pct_Over under 8% without compromising MAE excessively).

---

## Execution Steps

### 1. Validate Metrics

- Use `view_file` to read the tables inside `docs/results.md` and `README.md`.
- Ensure the new metrics justify an update.

### 2. Update docs/results.md

- Update the main header to the current date: `## 🏆 1. Kết Quả Thực Nghiệm (Cập nhật DD/MM/YYYY)`.
- Update the `Hiệu Năng Audit` table values for `MAE`, `Shortage %`, and `Oversupply %`.
- Add a new bullet point under `### ✅ Cải tiến Đột phá (Cập nhật mới)` briefly describing the technical change that led to this improvement.

### 3. Update README.md

- Update the section header date: `## 📈 Chỉ số Hiệu năng (Sau nâng cấp DD/MM/YYYY)`.
- Update the `Chỉ số Hiệu năng` table with the newly achieved results.

### 4. Version Control

- Run the bash command: `git add README.md docs/results.md`
- Run the bash command: `git commit -m "docs: update performance metrics to latest record based on workflow"`
- Run the bash command: `git push origin dev` (or current branch).

---

## Example Expected Output

```markdown
## 🚀 Docs Successfully Updated

### New Records Achieved 🏆

- **MAE:** 1,040 m³
- **Shortage:** 0.28%
- **Oversupply:** 6.99%

### Modified Files

- `README.md`
- `docs/results.md`
```
