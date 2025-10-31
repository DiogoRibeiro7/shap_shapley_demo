# SHAP Shapley Demo

This project demonstrates how to use **SHAP (SHapley Additive exPlanations)** to interpret a Random Forest classifier trained on the Breast Cancer dataset.

## Features
- Custom baseline value (expected prediction)
- Class probability decomposition
- Waterfall, summary, and dependence plots
- Exportable explanations for model auditing

## Structure
```
src/shap_explain.py      – Full runnable script
notebooks/shap_demo.ipynb – Interactive notebook with markdown explanations
```

## Run Example

```bash
pip install -r requirements.txt
python src/shap_explain.py
```
