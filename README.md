# Synthetic Project Management Analytics

This repository contains a fully reproducible analytics project built around a **synthetic project management** dataset.  It is designed to demonstrate the skills expected of business analysts, program managers and data analysts.  The project walks through exploratory data analysis, visualization and predictive modelling on a clean dataset, making it suitable as a portfolio piece when applying for data–focused roles.

## Dataset

The file `synthetic_project_data.csv` holds 1 000 rows of artificial project data.  Each row describes a fictional project with the following fields:

| Column | Description |
|------|-------------|
| `team_size` | Size of the project team (between 2 and 10 members). |
| `budget` | Budget allocated to the project in US dollars (between 50 000 and 1 000 000). |
| `complexity` | Project complexity on a five‑point scale (1 = lowest complexity, 5 = highest). |
| `client_importance` | Importance of the client on a five‑point scale (1 = lowest importance, 5 = highest). |
| `duration_days` | Actual duration of the project in days (30–364). |
| `success` | Binary label indicating whether the project was successful (1 = success, 0 = failure).  This outcome is generated using a logistic function of the other features to simulate realistic correlations. |

The dataset is **synthetic**, meaning it does not contain real personal or corporate data.  Using synthetic data is a common way to protect privacy while still enabling analytics and model development【865990695532403†L243-L304】.

## Requirements

All dependencies are listed in `requirements.txt`.  Install them with:

```bash
pip install -r requirements.txt
```

The key libraries used are:

- **pandas** and **numpy** for data manipulation.
- **matplotlib** and **seaborn** for visualization.
- **scikit‑learn** for building regression and classification models.

## Running the analysis

The primary analysis is contained in the Jupyter notebook `analysis_notebook.ipynb`.  The notebook is structured to increase in complexity as you progress:

1. **Introduction & Data Loading** – read the CSV into a pandas DataFrame and inspect the first few rows.
2. **Descriptive Statistics** – compute summary statistics to understand the range and distribution of each variable.
3. **Exploratory Visualisations** – plot histograms and a correlation heatmap to explore relationships among variables.
4. **Regression Modelling** – build a linear regression model to predict project duration (`duration_days`) from the other features.  Evaluate performance using mean absolute error and R².
5. **Classification Modelling** – build a logistic regression model to predict project success (`success`).  Evaluate performance using accuracy, a classification report and a confusion matrix.  This section demonstrates how logistic regression converts input features into probabilities using the sigmoid function【514564861997090†L80-L86】.
6. **Probability Visualisation** – visualise the distribution of predicted probabilities of success to interpret model confidence.

To run the notebook:

```bash
# Clone the repository
git clone <your‑fork‑url>
cd synthetic-project-management-analytics

# (Optional) create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter and open the notebook
jupyter notebook analysis_notebook.ipynb
```

Execute the cells in order; each section builds on the previous one.  Feel free to experiment by adding additional models (e.g. random forests, gradient boosting) or performing hyper‑parameter tuning.

## Regenerating the dataset

If you wish to customise the dataset (e.g. change the number of samples or adjust feature distributions), refer to the script `create_repo_project.py`.  It defines a function `generate_synthetic_dataset` that can be called to produce a new CSV file with customised parameters.  Regenerating the data is completely deterministic when you supply the same random seed, making it easy to reproduce results.

## Motivation

This project is intended as a portfolio example demonstrating:

- How to build a reproducible data pipeline using Python.
- How to perform exploratory data analysis and visualise key insights.
- How to apply both regression and classification models to real‑world style problems.
- How synthetic data can be used to safely practise analytic skills while avoiding privacy concerns【865990695532403†L243-L304】.

Please feel free to fork this repository, experiment with the notebook, and adapt the project to your own needs.  Contributions and suggestions are welcome!
