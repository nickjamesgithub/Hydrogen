from scipy.spatial.distance import correlation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dcor

def distance_correlation(a,b):
    return dcor.distance_correlation(a,b)

data = pd.read_csv("/Users/tassjames/Desktop/carbon_credits_research/hydrogen_research/Hydrogen_data.csv")
data['Capacity'] = pd.to_numeric(data['Capacity'])
data['Year'] = pd.to_numeric(data['Year'])
data_remove = data.dropna()

# # Remove fossil fuels
# data_remove = data_remove[data_remove.Tech != "Fossil"]

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

# List of cumulative counts for continents
cumulative_capacity_regions = np.array([europe_cum, east_asia_cum, north_america_cum, oceania_cum, south_america_cum, other_asia_cum])

distance_correlation_usa_europe_plants = [] # USA/Europe
distance_correlation_other_asia_europe_plants = [] # Other Asia/Europe
distance_correlation_other_asia_usa_plants = [] # Other Asia/USA

for i in range(5, len(years)):
    europe_slice = europe_cum[i-5:i]
    usa_slice = north_america_cum[i-5:i]
    other_asia_slice = other_asia_cum[i - 5:i]

    # DCORR USA/Europe
    dist_corr_usa_europe = dcor.distance_correlation(europe_slice,usa_slice)
    distance_correlation_usa_europe_plants.append(dist_corr_usa_europe)

    # DCORR Europe/Other Asia
    dist_corr_europe_other_asia = dcor.distance_correlation(europe_slice, other_asia_slice)
    distance_correlation_other_asia_europe_plants.append(dist_corr_europe_other_asia)

    # DCORR USA/Other Asia
    dist_corr_usa_other_asia = dcor.distance_correlation(usa_slice, other_asia_slice)
    distance_correlation_other_asia_usa_plants.append(dist_corr_usa_other_asia)

# plot of distance correlation
year_grid_plot = np.linspace(2005, 2024, len(distance_correlation_usa_europe_plants))

# Plot USA/Europe
plt.plot(year_grid_plot, distance_correlation_usa_europe_plants)
plt.ylabel("Distance correlation")
plt.xlabel("Time")
plt.show()

# Plot USA/East Asia
plt.plot(year_grid_plot, distance_correlation_other_asia_europe_plants)
plt.ylabel("Distance correlation")
plt.xlabel("Time")
plt.show()

# Plot Europe/East Asia
plt.plot(year_grid_plot, distance_correlation_other_asia_usa_plants)
plt.ylabel("Distance correlation")
plt.xlabel("Time")
plt.show()





