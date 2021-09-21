from scipy.spatial.distance import correlation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

# Import Data
data = pd.read_csv("/Users/tassjames/Desktop/carbon_credits_research/hydrogen_research/Hydrogen_data.csv")
data['Capacity'] = pd.to_numeric(data['Capacity'])
data['Year'] = pd.to_numeric(data['Year'])
data_remove = data.dropna()

# Generate grid of years
years = np.linspace(2000, 2024, 25)

# Capacity Country lists
europe_fossil_list_capacity = []
europe_green_list_capacity = []
# East Asia
east_asia_fossil_list_capacity = []
east_asia_green_list_capacity = []
# North America
north_america_fossil_list_capacity = []
north_america_green_list_capacity = []
# Oceania
oceania_fossil_list_capacity = []
oceania_green_list_capacity = []
# South America
south_america_fossil_list_capacity = []
south_america_green_list_capacity = []
# Other Asia
other_asia_fossil_list_capacity = []
other_asia_green_list_capacity = []

for j in range(len(years)):
    # Europe
    europe_fossil_capacity = data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'Europe')
                                                 & (data_remove['Tech'] == 'Fossil'), 'Capacity'].sum()
    europe_fossil_list_capacity.append(europe_fossil_capacity)

    europe_green_capacity = data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'Europe')
                                                 & (data_remove['Tech'] != 'Fossil'), 'Capacity'].sum()
    europe_green_list_capacity.append(europe_green_capacity)

    # East Asia
    east_asia_fossil_capacity = data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'East Asia')
                                                 & (data_remove['Tech'] == 'Fossil'), 'Capacity'].sum()
    east_asia_fossil_list_capacity.append(east_asia_fossil_capacity)

    east_asia_green_capacity = data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'East Asia')
                                                 & (data_remove['Tech'] != 'Fossil'), 'Capacity'].sum()
    east_asia_green_list_capacity.append(east_asia_green_capacity)

    # North America
    north_america_fossil_capacity = data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'North America')
                                                 & (data_remove['Tech'] == 'Fossil'), 'Capacity'].sum()
    north_america_fossil_list_capacity.append(north_america_fossil_capacity)

    north_america_green_capacity = data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'North America')
                                                 & (data_remove['Tech'] != 'Fossil'), 'Capacity'].sum()
    north_america_green_list_capacity.append(north_america_green_capacity)

    # Oceania
    oceania_fossil_capacity = data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'Oceania')
                                                 & (data_remove['Tech'] == 'Fossil'), 'Capacity'].sum()
    oceania_fossil_list_capacity.append(oceania_fossil_capacity)

    oceania_green_capacity = data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'Oceania')
                                                 & (data_remove['Tech'] != 'Fossil'), 'Capacity'].sum()
    oceania_green_list_capacity.append(oceania_green_capacity)

    # South America
    south_america_fossil_capacity = data_remove.loc[(data_remove['Year'] == years[j])
                                                  & (data_remove['Continent'] == 'South America')
                                                  & (data_remove['Tech'] == 'Fossil'), 'Capacity'].sum()
    south_america_fossil_list_capacity.append(south_america_fossil_capacity)

    south_america_green_capacity = data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'South America')
                                                 & (data_remove['Tech'] != 'Fossil'), 'Capacity'].sum()
    south_america_green_list_capacity.append(south_america_green_capacity)

    # Other Asia
    other_asia_fossil_capacity = data_remove.loc[(data_remove['Year'] == years[j])
                                                  & (data_remove['Continent'] == 'Other Asia')
                                                  & (data_remove['Tech'] == 'Fossil'), 'Capacity'].sum()
    other_asia_fossil_list_capacity.append(other_asia_fossil_capacity)

    other_asia_green_capacity = data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'Other Asia')
                                                 & (data_remove['Tech'] != 'Fossil'), 'Capacity'].sum()
    other_asia_green_list_capacity.append(other_asia_green_capacity)

# Cumulative Plants per continent: Green
europe_green_cum_capacity = np.cumsum(europe_green_list_capacity)
east_asia_green_cum_capacity = np.cumsum(east_asia_green_list_capacity)
north_america_green_cum_capacity = np.cumsum(north_america_green_list_capacity)
oceania_green_cum_capacity = np.cumsum(oceania_green_list_capacity)
south_america_green_cum_capacity = np.cumsum(south_america_green_list_capacity)
other_asia_green_cum_capacity = np.cumsum(other_asia_green_list_capacity)

# Cumulative Plants per continent: Fossil
europe_fossil_cum_capacity = np.cumsum(europe_fossil_list_capacity)
east_asia_fossil_cum_capacity = np.cumsum(east_asia_fossil_list_capacity)
north_america_fossil_cum_capacity = np.cumsum(north_america_fossil_list_capacity)
oceania_fossil_cum_capacity = np.cumsum(oceania_fossil_list_capacity)
south_america_fossil_cum_capacity = np.cumsum(south_america_fossil_list_capacity)
other_asia_fossil_cum_capacity = np.cumsum(other_asia_fossil_list_capacity)

# Capacity Fossil contribution
capacity_europe = np.nan_to_num(europe_fossil_cum_capacity/(europe_green_cum_capacity+europe_fossil_cum_capacity))
capacity_east_asia = np.nan_to_num(east_asia_fossil_cum_capacity/(east_asia_green_cum_capacity+east_asia_fossil_cum_capacity))
capacity_north_america = np.nan_to_num(north_america_fossil_cum_capacity/(north_america_green_cum_capacity+north_america_fossil_cum_capacity))
capacity_oceania = np.nan_to_num(oceania_fossil_cum_capacity/(oceania_green_cum_capacity+oceania_fossil_cum_capacity))
capacity_south_america = np.nan_to_num(south_america_fossil_cum_capacity/(south_america_green_cum_capacity+south_america_fossil_cum_capacity))
capacity_other_asia = np.nan_to_num(other_asia_fossil_cum_capacity/(other_asia_green_cum_capacity+other_asia_fossil_cum_capacity))


# Plants Country lists
europe_fossil_list_plants = []
europe_green_list_plants = []
# East Asia
east_asia_fossil_list_plants = []
east_asia_green_list_plants = []
# North America
north_america_fossil_list_plants = []
north_america_green_list_plants = []
# Oceania
oceania_fossil_list_plants = []
oceania_green_list_plants = []
# South America
south_america_fossil_list_plants = []
south_america_green_list_plants = []
# Other Asia
other_asia_fossil_list_plants = []
other_asia_green_list_plants = []

for j in range(len(years)):
    # Europe
    europe_fossil_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'Europe')
                                                 & (data_remove['Tech'] == 'Fossil')])
    europe_fossil_list_plants.append(europe_fossil_capacity)

    europe_green_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'Europe')
                                                 & (data_remove['Tech'] != 'Fossil')])
    europe_green_list_plants.append(europe_green_capacity)

    # East Asia
    east_asia_fossil_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'East Asia')
                                                 & (data_remove['Tech'] == 'Fossil')])
    east_asia_fossil_list_plants.append(east_asia_fossil_capacity)

    east_asia_green_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'East Asia')
                                                 & (data_remove['Tech'] != 'Fossil')])
    east_asia_green_list_plants.append(east_asia_green_capacity)

    # North America
    north_america_fossil_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'North America')
                                                 & (data_remove['Tech'] == 'Fossil')])
    north_america_fossil_list_plants.append(north_america_fossil_capacity)

    north_america_green_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'North America')
                                                 & (data_remove['Tech'] != 'Fossil')])
    north_america_green_list_plants.append(north_america_green_capacity)

    # Oceania
    oceania_fossil_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'Oceania')
                                                 & (data_remove['Tech'] == 'Fossil')])
    oceania_fossil_list_plants.append(oceania_fossil_capacity)

    oceania_green_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'Oceania')
                                                 & (data_remove['Tech'] != 'Fossil')])
    oceania_green_list_plants.append(oceania_green_capacity)

    # South America
    south_america_fossil_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                  & (data_remove['Continent'] == 'South America')
                                                  & (data_remove['Tech'] == 'Fossil')])
    south_america_fossil_list_plants.append(south_america_fossil_capacity)

    south_america_green_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'South America')
                                                 & (data_remove['Tech'] != 'Fossil')])
    south_america_green_list_plants.append(south_america_green_capacity)

    # Other Asia
    other_asia_fossil_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                  & (data_remove['Continent'] == 'Other Asia')
                                                  & (data_remove['Tech'] == 'Fossil')])
    other_asia_fossil_list_plants.append(other_asia_fossil_capacity)

    other_asia_green_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'Other Asia')
                                                 & (data_remove['Tech'] != 'Fossil')])
    other_asia_green_list_plants.append(other_asia_green_capacity)

# Cumulative Plants per continent: Green
europe_green_cum_plants = np.cumsum(europe_green_list_plants)
east_asia_green_cum_plants = np.cumsum(east_asia_green_list_plants)
north_america_green_cum_plants = np.cumsum(north_america_green_list_plants)
oceania_green_cum_plants = np.cumsum(oceania_green_list_plants)
south_america_green_cum_plants = np.cumsum(south_america_green_list_plants)
other_asia_green_cum_plants = np.cumsum(other_asia_green_list_plants)

# Cumulative Plants per continent: Fossil
europe_fossil_cum_plants = np.cumsum(europe_fossil_list_plants)
east_asia_fossil_cum_plants = np.cumsum(east_asia_fossil_list_plants)
north_america_fossil_cum_plants = np.cumsum(north_america_fossil_list_plants)
oceania_fossil_cum_plants = np.cumsum(oceania_fossil_list_plants)
south_america_fossil_cum_plants = np.cumsum(south_america_fossil_list_plants)
other_asia_fossil_cum_plants = np.cumsum(other_asia_fossil_list_plants)

# Capacity Fossil contribution
plants_europe = np.nan_to_num(europe_fossil_cum_plants/(europe_green_cum_plants+europe_fossil_cum_plants))
plants_east_asia = np.nan_to_num(east_asia_fossil_cum_plants/(east_asia_green_cum_plants+east_asia_fossil_cum_plants))
plants_north_america = np.nan_to_num(north_america_fossil_cum_plants/(north_america_green_cum_plants+north_america_fossil_cum_plants))
plants_oceania = np.nan_to_num(oceania_fossil_cum_plants/(oceania_green_cum_plants+oceania_fossil_cum_plants))
plants_south_america = np.nan_to_num(south_america_fossil_cum_plants/(south_america_green_cum_plants+south_america_fossil_cum_plants))
plants_other_asia = np.nan_to_num(other_asia_fossil_cum_plants/(other_asia_green_cum_plants+other_asia_fossil_cum_plants))

#todo fix the error with the 1's / 0's 
# Cross Contextual Analysis
consistency_norm_list = []
for i in range(len(plants_europe)):
    # Get plant numbers and capacity
    plants_t = np.array([plants_europe[i], plants_east_asia[i], plants_north_america[i], plants_oceania[i], plants_south_america[i], plants_other_asia[i]]).reshape(-1,1)
    capacity_t = np.array([capacity_europe[i], capacity_east_asia[i], capacity_north_america[i], capacity_oceania[i], capacity_south_america[i], capacity_other_asia[i]]).reshape(-1,1)

    # Plants and capacity distance matrices
    plants_t_dist = manhattan_distances(plants_t, plants_t)
    capacity_t_dist = manhattan_distances(capacity_t, capacity_t)

    # Convert to affinity matrices
    affinity_plants = 1 - plants_t_dist/np.max(plants_t_dist)
    affinity_capacity = 1 - capacity_t_dist / np.max(capacity_t_dist)

    # Consistency matrix norm
    consistency_norm = np.nan_to_num(np.sum(np.abs(affinity_plants - affinity_capacity)))
    consistency_norm_list.append(consistency_norm)

# Plot consistency matrix norm
plt.plot(consistency_norm_list)
plt.ylabel("Consistency matrix norm")
plt.xlabel("Time")
plt.savefig("CCA_capacity_plants")
plt.show()