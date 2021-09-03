from scipy.spatial.distance import correlation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Function to calculate the power-law with constants a and b
def power_law(x, a, b):
    return a*np.power(x, b)

def exponential(x, a, b):
    return a*np.exp(b*x)

data = pd.read_csv("/Users/tassjames/Desktop/carbon_credits_research/hydrogen_research/Hydrogen_data.csv")
data['Capacity'] = pd.to_numeric(data['Capacity'])
data['Year'] = pd.to_numeric(data['Year'])
data_remove = data.dropna()

# Remove fossil fuels
data_remove = data_remove[data_remove.Tech != "Fossil"]

# Generate grid of years
years = np.linspace(2000, 2024, 25)

# Country lists
europe_list = []
east_asia_list = []
north_america_list = []
oceania_list = []
south_america_list = []
other_asia_list = []
for j in range(len(years)):
    # Europe
    europe_capacity_yearly = data_remove.loc[(data_remove['Year'] == years[j]) & (data_remove['Continent'] == 'Europe'), 'Capacity'].sum()
    europe_list.append(europe_capacity_yearly)
    # East Asia
    east_asia_capacity_yearly = data_remove.loc[(data_remove['Year'] == years[j]) & (data_remove['Continent'] == 'East Asia'), 'Capacity'].sum()
    east_asia_list.append(east_asia_capacity_yearly)
    # North America
    north_america_capacity_yearly = data_remove.loc[(data_remove['Year'] == years[j]) & (data_remove['Continent'] == 'North America'), 'Capacity'].sum()
    north_america_list.append(north_america_capacity_yearly)
    # Oceania
    oceania_capacity_yearly = data_remove.loc[(data_remove['Year'] == years[j]) & (data_remove['Continent'] == 'Oceania'), 'Capacity'].sum()
    oceania_list.append(oceania_capacity_yearly)
    # South America
    south_america_capacity_yearly = data_remove.loc[(data_remove['Year'] == years[j]) & (data_remove['Continent'] == 'South America'), 'Capacity'].sum()
    south_america_list.append(south_america_capacity_yearly)
    # other Asia
    other_asia_capacity_yearly = data_remove.loc[(data_remove['Year'] == years[j]) & (data_remove['Continent'] == 'Other Asia'), 'Capacity'].sum()
    other_asia_list.append(other_asia_capacity_yearly)

# Cumulative counts for each region
europe_cum = np.cumsum(europe_list)
east_asia_cum = np.cumsum(east_asia_list)
north_america_cum = np.cumsum(north_america_list)
oceania_cum = np.cumsum(oceania_list)
south_america_cum = np.cumsum(south_america_list)
other_asia_cum = np.cumsum(other_asia_list)

# Plot the raw data (no estimation/optimisation)
plt.scatter(years, europe_cum, label="Europe")
# plt.scatter(years, east_asia_cum, label="East Asia")
# plt.scatter(years, north_america_cum, label="North America")
# plt.scatter(years, oceania_cum, label="Oceania")
# plt.scatter(years, south_america_cum, label="South America")
# plt.scatter(years, other_asia_cum, label="other asia")
plt.legend()
plt.show()

# Fit the dummy power-law data
europe_pars, europe_cov = curve_fit(f=power_law, xdata=years, ydata=south_america_cum, p0=[0, 0], bounds=(-np.inf, np.inf)) # np.inf (bounds)
# Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
europe_stdevs = np.sqrt(np.diag(europe_cov))
# Calculate the residuals
europe_res = europe_cum - power_law(years, *europe_pars)

# Plot power law
plt.plot(europe_cum)
plt.plot(power_law(years, *europe_pars))
plt.show()

