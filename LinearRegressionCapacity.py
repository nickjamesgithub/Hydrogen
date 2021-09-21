import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data
data = pd.read_csv("/Users/tassjames/Desktop/carbon_credits_research/hydrogen_research/Hydrogen_data.csv")
data['Capacity'] = pd.to_numeric(data['Capacity'])
data['Year'] = pd.to_numeric(data['Year'])

# Fill N/A with 0
# Remove rows with N/A values
data_remove = data.dropna()
data_remove['log_capacity'] = np.log(data_remove['Capacity'])

# Fit OLS model
model_basic = smf.ols(formula='log_capacity ~ Continent + Tech + Year', data=data_remove)
result = model_basic.fit()
print(result.summary())

