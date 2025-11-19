pip install pandas numpy matplotlib seaborn ipywidgets
# If using Jupyter Notebook/Lab, also enable widgets:
jupyter nbextension enable --py widgetsnbextension
# For JupyterLab you may need the labextension:
# jupyter labextension install @jupyter-widgets/jupyterlab-manager

# analysis.py
# Author contact (required): 23f2000416@ds.study.iitm.ac.in
#
# This file is formatted as a Jupyter/VS Code "Python script" notebook (uses `# %%` cell markers).
# Run it in JupyterLab, VS Code (Python extension), or any environment that supports ipywidgets.
#
# Purpose: interactive exploration of relationships between synthetic supply-chain variables.
# The slider controls how many recent rows are included in the analysis; downstream cells depend
# on the selected subset and recompute statistics and plots dynamically.

# %% [markdown]
# # Interactive Data Analysis Notebook (Marimo-style)
# This interactive notebook demonstrates relationships between variables and updates dynamically
# when the slider value changes. The slider selects how many most-recent rows of the dataset are used.

# %%
# CELL 1 — Data generation / loading and shared variables
# Data flow note: This cell *creates* the main dataframe `df_all` and a helper function
# `get_recent_df(n)` that downstream cells call to obtain the subset controlled by the slider.
import numpy as np
import pandas as pd

# Seed for reproducibility
RND = 42
np.random.seed(RND)

# Create a synthetic dataset representing supply chain metrics (74 rows like your assignment)
n_rows = 74
df_all = pd.DataFrame({
    "Supplier_Lead_Time": np.clip(np.random.normal(loc=12, scale=5, size=n_rows), 1, None),  # days
    "Inventory_Levels": np.clip(np.random.normal(loc=300, scale=90, size=n_rows), 0, None),  # units
    "Order_Frequency": np.clip(np.random.normal(loc=6, scale=2.5, size=n_rows), 0.5, None),  # per month
    "Delivery_Performance": np.clip(np.random.normal(loc=85, scale=8, size=n_rows), 40, 100), # %
    "Cost_Per_Unit": np.clip(np.random.normal(loc=28, scale=7, size=n_rows), 1, None)        # $
})

# show top rows (if run in a notebook cell, this prints a table)
df_all.head()

# Helper function to get the most recent n rows (downstream code depends on this)
def get_recent_df(n_rows_to_use):
    """
    Returns the last n_rows_to_use rows from df_all.
    Downstream cells call this function whenever the slider changes.
    """
    if n_rows_to_use < 2:
        # minimal safe size for correlation computation
        n_rows_to_use = 2
    return df_all.tail(int(n_rows_to_use)).reset_index(drop=True)

# %% [markdown]
# ## Instructions
# - Use the slider below to choose how many of the most recent rows are used in the analysis.
# - The summary statistics, correlation matrix and dynamic markdown update automatically.

# %%
# CELL 2 — Interactive widget and dynamic output
# Data flow note: This cell *reads* from get_recent_df(n) to compute stats and render dynamic markdown.
# Variables created here (e.g., current_df, corr) reflect the slider state and are not global constants.

from ipywidgets import IntSlider, VBox, HBox, interactive_output
from IPython.display import display, Markdown, clear_output
import matplotlib.pyplot as plt
import seaborn as sns

# Create slider: choose subset size from 5 to full dataset size
slider = IntSlider(value=20, min=5, max=len(df_all), step=1, description='Rows:', continuous_update=True)

def render_analysis(n_rows_selected):
    """
    Callback invoked on slider change.
    Recomputes the subset, summary, correlation matrix and updates dynamic markdown + plots.
    """
    # Acquire data subset (dependency on cell 1)
    current_df = get_recent_df(n_rows_selected)

    # Compute simple summary stats
    summary = current_df.describe().loc[['mean','std','min','50%','max']].T.round(2)

    # Compute correlation matrix
    corr = current_df.corr().round(3)

    # Clear previous output area and render new markdown and visuals
    clear_output(wait=True)

    # Dynamic markdown: shows data usage and a short interpretation
    md = f"""
### Interactive analysis (using last **{n_rows_selected}** rows)

**Observations (dynamically computed):**
- Mean Supplier Lead Time: **{summary.loc['Supplier_Lead_Time','mean']}** days  
- Mean Inventory Level: **{summary.loc['Inventory_Levels','mean']}** units  
- Delivery Performance (mean): **{summary.loc['Delivery_Performance','mean']}** %

**Top correlations (absolute value):**
"""
    # derive top pairs (excluding self-correlation)
    corr_unstack = corr.unstack().reset_index()
    corr_unstack.columns = ['var1','var2','corr']
    # filter var1 < var2 to avoid duplicates
    corr_pairs = corr_unstack[corr_unstack['var1'] < corr_unstack['var2']].copy()
    corr_pairs['abs_corr'] = corr_pairs['corr'].abs()
    top_pairs = corr_pairs.sort_values('abs_corr', ascending=False).head(3)

    for _, row in top_pairs.iterrows():
        md += f"- **{row['var1']}** vs **{row['var2']}**: correlation = **{row['corr']}**  \n"

    # Display dynamic markdown
    display(Markdown(md))

    # Display the numeric correlation matrix
    display(Markdown("**Correlation matrix:**"))
    display(corr)

    # Visual heatmap (small)
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1, vmax=1, square=True)
    plt.title(f"Correlation Heatmap — last {n_rows_selected} rows")
    plt.show()

    # Show the summary table too
    display(Markdown("**Summary statistics (selected subset):**"))
    display(summary)

# Connect the slider to the rendering function
out = interactive_output(render_analysis, {'n_rows_selected': slider})

# Layout
display(VBox([slider, out]))
