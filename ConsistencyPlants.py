from scipy.spatial.distance import correlation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/tassjames/Desktop/carbon_credits_research/hydrogen_research/Hydrogen_data.csv")
data['Capacity'] = pd.to_numeric(data['Capacity'])
data['Year'] = pd.to_numeric(data['Year'])
data_remove = data.dropna()

# # Remove fossil fuels
# data_remove = data_remove[data_remove.Tech != "Fossil"]

# Generate grid of years
years = np.linspace(2000, 2024, 25)

# Country lists
europe_fossil_list = []
europe_green_list = []
# East Asia
east_asia_fossil_list = []
east_asia_green_list = []
# North America
north_america_fossil_list = []
north_america_green_list = []
# Oceania
oceania_fossil_list = []
oceania_green_list = []
# South America
south_america_fossil_list = []
south_america_green_list = []
# Other Asia
other_asia_fossil_list = []
other_asia_green_list = []

for j in range(len(years)):
    # Europe
    europe_fossil_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'Europe')
                                                 & (data_remove['Tech'] == 'Fossil')])
    europe_fossil_list.append(europe_fossil_capacity)

    europe_green_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'Europe')
                                                 & (data_remove['Tech'] != 'Fossil')])
    europe_green_list.append(europe_green_capacity)

    # East Asia
    east_asia_fossil_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'East Asia')
                                                 & (data_remove['Tech'] == 'Fossil')])
    east_asia_fossil_list.append(east_asia_fossil_capacity)

    east_asia_green_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'East Asia')
                                                 & (data_remove['Tech'] != 'Fossil')])
    east_asia_green_list.append(east_asia_green_capacity)

    # North America
    north_america_fossil_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'North America')
                                                 & (data_remove['Tech'] == 'Fossil')])
    north_america_fossil_list.append(north_america_fossil_capacity)

    north_america_green_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'North America')
                                                 & (data_remove['Tech'] != 'Fossil')])
    north_america_green_list.append(north_america_green_capacity)

    # Oceania
    oceania_fossil_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'Oceania')
                                                 & (data_remove['Tech'] == 'Fossil')])
    oceania_fossil_list.append(oceania_fossil_capacity)

    oceania_green_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'Oceania')
                                                 & (data_remove['Tech'] != 'Fossil')])
    oceania_green_list.append(oceania_green_capacity)

    # South America
    south_america_fossil_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                  & (data_remove['Continent'] == 'South America')
                                                  & (data_remove['Tech'] == 'Fossil')])
    south_america_fossil_list.append(south_america_fossil_capacity)

    south_america_green_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'South America')
                                                 & (data_remove['Tech'] != 'Fossil')])
    south_america_green_list.append(south_america_green_capacity)

    # Other Asia
    other_asia_fossil_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                  & (data_remove['Continent'] == 'Other Asia')
                                                  & (data_remove['Tech'] == 'Fossil')])
    other_asia_fossil_list.append(other_asia_fossil_capacity)

    other_asia_green_capacity = len(data_remove.loc[(data_remove['Year'] == years[j])
                                                 & (data_remove['Continent'] == 'Other Asia')
                                                 & (data_remove['Tech'] != 'Fossil')])
    other_asia_green_list.append(other_asia_green_capacity)

# Cumulative Plants per continent: Green
europe_green_cum = np.cumsum(europe_green_list)
east_asia_green_cum = np.cumsum(east_asia_green_list)
north_america_green_cum = np.cumsum(north_america_green_list)
oceania_green_cum = np.cumsum(oceania_green_list)
south_america_green_cum = np.cumsum(south_america_green_list)
other_asia_green_cum = np.cumsum(other_asia_green_list)

# Cumulative Plants per continent: Fossil
europe_fossil_cum = np.cumsum(europe_fossil_list)
east_asia_fossil_cum = np.cumsum(east_asia_fossil_list)
north_america_fossil_cum = np.cumsum(north_america_fossil_list)
oceania_fossil_cum = np.cumsum(oceania_fossil_list)
south_america_fossil_cum = np.cumsum(south_america_fossil_list)
other_asia_fossil_cum = np.cumsum(other_asia_fossil_list)

# Green and fossil cumulative
green_cumulative = np.array([europe_green_cum, east_asia_green_cum, north_america_green_cum, oceania_green_cum, south_america_green_cum, other_asia_green_cum])
fossil_cumulative = np.array([europe_fossil_cum, east_asia_fossil_cum, north_america_fossil_cum, oceania_fossil_cum, south_america_fossil_cum, other_asia_fossil_cum])

# Names of continents
names = ["Europe", "East Asia", "North America", "Oceania", "South America", "Other Asia"]

# Append % Fossil up
for i in range(len(green_cumulative)):
    fossil_percentage = fossil_cumulative[i]/(fossil_cumulative[i]+green_cumulative[i])

    # Plot
    plt.plot(years, fossil_percentage, label=names[i])
    plt.title("Number of plants")
    plt.xlabel("Time")
    plt.ylabel("Proportion of fossil plants")
    plt.ylim(0,1)
    plt.legend()
plt.savefig("Fossil_green_ratio_number_plants")
plt.show()