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


## Next Steps

**Done.** Weekly stakeholder report automation is implemented — see ### Implemented below.

---
*Analytics portfolio project — 2025-08*

<!-- build 6 -->

### Implemented

```bash
pip install -r requirements.txt
python scripts/weekly_report.py
```