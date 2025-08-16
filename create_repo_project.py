import pandas as pd
import numpy as np
import nbformat as nbf

# Create synthetic dataset
def generate_synthetic_dataset(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    team_size = np.random.randint(2, 11, size=n_samples)  # team sizes from 2-10
    budget = np.random.uniform(50000, 1000000, size=n_samples)  # budgets between 50k and 1M
    complexity = np.random.randint(1, 6, size=n_samples)  # complexity on scale 1-5
    client_importance = np.random.randint(1, 6, size=n_samples)  # importance 1-5
    duration_days = np.random.randint(30, 365, size=n_samples)  # duration in days

    # Compute success probability based on features
    # Higher team size, higher complexity and client importance, lower duration, higher budget increase success
    intercept = -5
    z = (
        intercept
        + 0.3 * team_size
        + 0.7 * complexity
        + 0.5 * client_importance
        - 0.01 * duration_days
        + 1e-6 * budget
    )
    prob_success = 1 / (1 + np.exp(-z))
    success = (np.random.rand(n_samples) < prob_success).astype(int)

    df = pd.DataFrame(
        {
            "team_size": team_size,
            "budget": budget,
            "complexity": complexity,
            "client_importance": client_importance,
            "duration_days": duration_days,
            "success": success,
        }
    )
    return df

# Save dataset
def save_dataset(df, path):
    df.to_csv(path, index=False)

# Create Jupyter notebook for analysis
def create_analysis_notebook(df_path, notebook_path):
    nb = nbf.v4.new_notebook()
    cells = []

    # Introduction and project overview
    intro_text = """
# Synthetic Project Management Dataset

This project demonstrates end-to-end data analysis on a synthetic project management dataset. 
The dataset contains information about fictitious projects including team size, budget, complexity, client importance, and duration. 
The goal is to explore the data, visualize relationships, and build predictive models to understand the factors impacting project success.

We will start with exploratory data analysis (EDA) using descriptive statistics and visualizations, and then progress to predictive modeling. 
We'll implement both regression and classification models to predict project duration and success, respectively.
"""
    cells.append(nbf.v4.new_markdown_cell(intro_text))

    # Load data
    code_load = f"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('{df_path}')

# Display the first few rows
df.head()
"""
    cells.append(nbf.v4.new_code_cell(code_load))

    # Basic statistics
    code_stats = """
# Summary statistics
df.describe()
"""
    cells.append(nbf.v4.new_code_cell(code_stats))

    # Visualize distributions
    code_visual1 = """
# Histogram for numeric features
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

numeric_cols = ['team_size', 'budget', 'complexity', 'client_importance', 'duration_days']

for idx, col in enumerate(numeric_cols):
    ax = axes[idx//3, idx%3]
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()
"""
    cells.append(nbf.v4.new_code_cell(code_visual1))

    # Correlation heatmap
    code_corr = """
# Correlation heatmap
plt.figure(figsize=(8, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()
"""
    cells.append(nbf.v4.new_code_cell(code_corr))

    # Predicting duration with Linear Regression
    code_regression = """
# Predicting project duration (regression)

# Features and target for regression
X_reg = df[['team_size', 'budget', 'complexity', 'client_importance']]
y_reg = df['duration_days']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Initialize and train model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict on test data
y_pred = lin_reg.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Model Performance:")
print(f"  Mean Absolute Error: {mae:.2f}")
print(f"  R-squared: {r2:.2f}")
"""
    cells.append(nbf.v4.new_code_cell(code_regression))

    # Predicting success with Logistic Regression
    code_classification = """
# Predicting project success (classification)

# Features and target for classification
X_clf = df[['team_size', 'budget', 'complexity', 'client_importance', 'duration_days']]
y_clf = df['success']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Initialize and train model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict on test data
y_pred = log_reg.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Logistic Regression Model Performance:")
print(f"  Accuracy: {acc:.2f}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", cm)
"""
    cells.append(nbf.v4.new_code_cell(code_classification))

    # Visualize predicted probabilities
    code_probabilities = """
# Plot predicted probabilities

# Fit model on entire dataset for probability visualization
log_reg_full = LogisticRegression(max_iter=1000)
log_reg_full.fit(X_clf, y_clf)

# Predict probabilities
probabilities = log_reg_full.predict_proba(X_clf)[:, 1]

plt.figure(figsize=(8, 5))
sns.histplot(probabilities, bins=30, kde=True)
plt.title('Distribution of Predicted Success Probabilities')
plt.xlabel('Probability of Success')
plt.ylabel('Frequency')
plt.show()
"""
    cells.append(nbf.v4.new_code_cell(code_probabilities))

    nb['cells'] = cells
    # Write to file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    df = generate_synthetic_dataset()
    dataset_path = '/home/oai/share/synthetic_project_data.csv'
    save_dataset(df, dataset_path)
    notebook_path = '/home/oai/share/analysis_notebook.ipynb'
    create_analysis_notebook(dataset_path, notebook_path)
    print('Dataset and notebook created.')
