from scipy.spatial.distance import correlation
import pandas as pd
import numpy as np
from pyemd import emd, emd_with_flow
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt

# Read in dataset
data = pd.read_csv("/Users/tassjames/Desktop/carbon_credits_research/hydrogen_research/Hydrogen_data.csv")
data['Capacity'] = pd.to_numeric(data['Capacity'])
data['Year'] = pd.to_numeric(data['Year'])
data_clean = data.dropna()

# Read in location data
location = pd.read_excel("/Users/tassjames/Desktop/carbon_credits_research/hydrogen_research/latitude_longitude_continents_updated.xlsx")

# Generate grid of years
years = np.linspace(2000, 2024, 25)
years_grid = np.linspace(2004, 2024, 21)
geodesic_variance = []

europe_list = []  # Average Capacity
east_asia_list = []
north_america_list = []
oceania_list = []
south_america_list = []
other_asia_list = []

for j in range(len(years)):
    # Log Europe
    log_europe_capacity = (data_clean.loc[(data_clean['Year'] == years[j]) &
                                        (data_clean['Continent'] == 'Europe'), 'Capacity'].sum())
    europe_list.append(log_europe_capacity)
    # Log East Asia
    log_east_asia_capacity = (data_clean.loc[(data_clean['Year'] == years[j]) &
                                                (data_clean['Continent'] == 'East Asia'), 'Capacity'].sum())
    east_asia_list.append(log_east_asia_capacity)
    # Log North America
    log_north_america_capacity = (data_clean.loc[(data_clean['Year'] == years[j]) &
                                                (data_clean['Continent'] == 'North America'), 'Capacity'].sum())
    north_america_list.append(log_north_america_capacity)
    # Log Oceania
    log_oceania_capacity = (data_clean.loc[(data_clean['Year'] == years[j]) &
                                                       (data_clean['Continent'] == 'Oceania'), 'Capacity'].sum())
    oceania_list.append(log_oceania_capacity)
    # Log South America
    log_south_america_capacity = (data_clean.loc[(data_clean['Year'] == years[j]) &
                                                       (data_clean['Continent'] == 'South America'), 'Capacity'].sum())
    south_america_list.append(log_south_america_capacity)
    # Log Other Asia
    log_other_asia_capacity = (data_clean.loc[(data_clean['Year'] == years[j]) &
                                                       (data_clean['Continent'] == 'Other Asia'), 'Capacity'].sum())
    other_asia_list.append(log_other_asia_capacity)

# Plot Europe list
plt.plot(europe_list, label="Europe")
plt.plot(east_asia_list, label="East Asia")
plt.plot(north_america_list, label="North America")
plt.plot(oceania_list, label="Oceania")
plt.plot(south_america_list, label="South America")
plt.plot(other_asia_list, label="Other Asia")
plt.legend()
plt.show()