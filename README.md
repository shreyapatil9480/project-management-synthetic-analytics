[![Python CI](https://github.com/shreyapatil9480/project-management-synthetic-analytics/actions/workflows/python-ci.yml/badge.svg)](https://github.com/shreyapatil9480/project-management-synthetic-analytics/actions/workflows/python-ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC)

# Project Management Synthetic Analytics

Which projects will deliver on time?

**Stakeholder:** PMO Lead

## Key Insights

- Scope changes above 4 per sprint correlate with late delivery.
- Teams above 85% load miss deadlines 22% more often.
- Velocity above 30 story points improves on-time delivery modestly.

## Dataset

Primary file: `data/project_delivery.csv`  
Target variable: `on_time`

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook notebooks/eda.ipynb
```



## Testing

```bash
pip install -r requirements.txt
pytest tests/ --cov=src
```

## Next Steps

Automate SQL exports into a weekly stakeholder report.

---
*Analytics portfolio project — 2025-08*

<!-- build 6 -->

### Implemented

```bash
pip install -r requirements.txt
python scripts/weekly_report.py
```
