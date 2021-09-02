import statsmodels.formula.api as smf
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import re
import statsmodels.api as sm
import pandas as pd

data = pd.read_csv("/Users/tassjames/Desktop/carbon_credits_research/hydrogen_research/Hydrogen_data.csv")
data['Capacity'] = pd.to_numeric(data['Capacity'])

# Fill N/A with 0
# data_fill = data.fillna(0)
# data_fill['log_capacity'] = np.nan_to_num(np.log(data_fill['Capacity']))
# Remove rows with N/A values
data_remove = data.dropna()
data_remove['log_capacity'] = np.log(data_remove['Capacity'])

# Fit OLS model
model_basic = smf.ols(formula='log_capacity ~ Continent + Tech + Year', data=data_remove)
result = model_basic.fit()
print(result.summary())

block = 1

