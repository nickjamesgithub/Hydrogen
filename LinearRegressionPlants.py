from scipy.spatial.distance import correlation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Function to calculate the intercept and slope of the linear function
def linear_model(x, a, b):
    return a*x + b

data = pd.read_csv("/Users/tassjames/Desktop/carbon_credits_research/hydrogen_research/Hydrogen_data.csv")
data['Capacity'] = pd.to_numeric(data['Capacity'])
data['Year'] = pd.to_numeric(data['Year'])
data_remove = data
# data_remove = data.dropna()

# # Remove fossil fuels
# data_remove = data_remove[data_remove.Tech != "Fossil"]

# Generate grid of years
years = np.linspace(2000, 2024, 25)
years_grid = np.linspace(1,25,25)

# Country lists
europe_list = []
east_asia_list = []
north_america_list = []
oceania_list = []
south_america_list = []
other_asia_list = []
for j in range(len(years)):
    # Europe
    europe_capacity_yearly = len(data_remove.loc[(data_remove['Year'] == years[j]) & (data_remove['Continent'] == 'Europe')])
    europe_list.append(europe_capacity_yearly)
    # East Asia
    east_asia_capacity_yearly = len(data_remove.loc[(data_remove['Year'] == years[j]) & (data_remove['Continent'] == 'East Asia')])
    east_asia_list.append(east_asia_capacity_yearly)
    # North America
    north_america_capacity_yearly = len(data_remove.loc[(data_remove['Year'] == years[j]) & (data_remove['Continent'] == 'North America')])
    north_america_list.append(north_america_capacity_yearly)
    # Oceania
    oceania_capacity_yearly = len(data_remove.loc[(data_remove['Year'] == years[j]) & (data_remove['Continent'] == 'Oceania')])
    oceania_list.append(oceania_capacity_yearly)
    # South America
    south_america_capacity_yearly = len(data_remove.loc[(data_remove['Year'] == years[j]) & (data_remove['Continent'] == 'South America')])
    south_america_list.append(south_america_capacity_yearly)
    # other Asia
    other_asia_capacity_yearly = len(data_remove.loc[(data_remove['Year'] == years[j]) & (data_remove['Continent'] == 'Other Asia')])
    other_asia_list.append(other_asia_capacity_yearly)

# Cumulative counts for each region
europe_cum = np.cumsum(europe_list)
east_asia_cum = np.cumsum(east_asia_list)
north_america_cum = np.cumsum(north_america_list)
oceania_cum = np.cumsum(oceania_list)
south_america_cum = np.cumsum(south_america_list)
other_asia_cum = np.cumsum(other_asia_list)

# Cumulative data
plt.plot(years, europe_cum, label="Europe")
plt.plot(years, east_asia_cum, label="East Asia")
plt.plot(years, north_america_cum, label="North America")
plt.plot(years, oceania_cum, label="Oceania")
plt.plot(years, south_america_cum, label="South America")
plt.plot(years, other_asia_cum, label="Other Asia")
plt.legend()
plt.savefig("Cumulative_plants")
plt.show()

# Names of continents
names = ["Europe", "East Asia", "North America", "Oceania", "South America", "Other Asia"]
cumulative_plants = np.array([europe_cum, east_asia_cum, north_america_cum, oceania_cum, south_america_cum, other_asia_cum])
for i in range(len(cumulative_plants)):
    # Generate data and name
    data = cumulative_plants[i]
    name = names[i]

    # Fit the linear model
    pars, cov = curve_fit(f=linear_model, xdata=years_grid, ydata=data, p0=[0, 0], bounds=(-np.inf, np.inf))
    # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    # Calculate the residuals
    res = data - linear_model(years_grid, *pars)
    print(res)

    # Plot linear regression estimate
    plt.scatter(years, data, label="Cumulative plants", color='black', alpha=0.5)
    plt.plot(years, linear_model(years_grid, *pars), label="Linear estimate", color='blue', linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Number of plants")
    plt.title(name)
    plt.legend()
    plt.savefig("linear_fit_"+name)
    plt.show()


