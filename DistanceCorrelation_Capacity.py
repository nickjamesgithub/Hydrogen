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

# List of cumulative counts for continents
cumulative_capacity_regions = np.array([europe_cum, east_asia_cum, north_america_cum, oceania_cum, south_america_cum, other_asia_cum])

distance_correlation_capacity = np.zeros((len(cumulative_capacity_regions), len(cumulative_capacity_regions)))
for i in range(len(cumulative_capacity_regions)):
    for j in range(len(cumulative_capacity_regions)):
        ts_i = cumulative_capacity_regions[i]
        ts_j = cumulative_capacity_regions[j]
        dist_corr = dcor.distance_correlation(ts_i,ts_j)
        distance_correlation_capacity[i,j] = dist_corr

# plot of distance correlation
plt.matshow(distance_correlation_capacity)
plt.colorbar()
plt.savefig("DCORR_Capacity")
plt.show()

# Distance correlation
print("Distance correlation matrix", distance_correlation_capacity)
print("Median distance correlation value", np.median(distance_correlation_capacity))

# Compute total distance correlation capacity
names = ["Europe", "East Asia", "North America", "Oceania", "South America", "Other Asia"]
for i in range(len(distance_correlation_capacity)):
    total_dist_correl = np.sum(distance_correlation_capacity[i])
    print(names[i], total_dist_correl)
