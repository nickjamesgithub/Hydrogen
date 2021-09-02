from scipy.spatial.distance import correlation
import pandas as pd
import numpy as np

data = pd.read_csv("/Users/tassjames/Desktop/carbon_credits_research/hydrogen_research/Hydrogen_data.csv")
data['Capacity'] = pd.to_numeric(data['Capacity'])
data['Year'] = pd.to_numeric(data['Year'])
data_remove = data.dropna()

# Generate grid of years
years = np.linspace(2000, 2028, 29)

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

block = 1


