from scipy.spatial.distance import correlation
import pandas as pd
import numpy as np
from pyemd import emd, emd_with_flow
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt

green = "yes"

# Read in dataset
data = pd.read_csv("/Users/tassjames/Desktop/carbon_credits_research/hydrogen_research/Hydrogen_data.csv")
data['Capacity'] = pd.to_numeric(data['Capacity'])
data['Year'] = pd.to_numeric(data['Year'])
data_remove = data.dropna()

# Remove fossil fuels from the data
if green:
    data_remove = data_remove[data_remove.Tech != "Fossil"]

# Read in location data
location = pd.read_excel("/Users/tassjames/Desktop/carbon_credits_research/hydrogen_research/latitude_longitude_continents_updated.xlsx")

# Write function for haversine distance
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 3956  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

# Read in latitude and longitude data
lats_array = np.array(location['Latitude'])
longs_array = np.array(location['Longitude'])
geographic_distance_matrix = np.zeros((len(location), len(location)))
for i in range(len(lats_array)):
    for j in range(len(longs_array)):
        lats_i = lats_array[i]
        lats_j = lats_array[j]
        longs_i = longs_array[i]
        longs_j = longs_array[j]
        distance = haversine(longs_i, lats_i, longs_j, lats_j)
        geographic_distance_matrix[i, j] = distance

# Generate grid of years
years = np.linspace(2000, 2024, 25)
geodesic_variance = []

# 5 year rolling Output
rolling_capacity = 4 # This is Python indexing
europe_list = []  # Average Capacity
for j in range(rolling_capacity, len(years)):
    europe_capacity_5 = data_remove.loc[(data_remove['Year'] >= years[j - rolling_capacity]) &
                                        (data_remove['Year'] <= years[j]) &
                                        (data_remove['Continent'] == 'Europe'), 'Capacity'].sum()
    east_asia_capacity_5 = data_remove.loc[(data_remove['Year'] >= years[j - rolling_capacity]) &
                                           (data_remove['Year'] <= years[j]) &
                                           (data_remove['Continent'] == 'East Asia'), 'Capacity'].sum()
    north_america_capacity_5 = data_remove.loc[(data_remove['Year'] >= years[j - rolling_capacity]) &
                                               (data_remove['Year'] <= years[j]) &
                                               (data_remove['Continent'] == 'North America'), 'Capacity'].sum()
    oceania_capacity_5 = data_remove.loc[(data_remove['Year'] >= years[j - rolling_capacity]) &
                                         (data_remove['Year'] <= years[j]) &
                                         (data_remove['Continent'] == 'Oceania'), 'Capacity'].sum()
    south_america_capacity_5 = data_remove.loc[(data_remove['Year'] >= years[j - rolling_capacity]) &
                                               (data_remove['Year'] <= years[j]) &
                                               (data_remove['Continent'] == 'South America'), 'Capacity'].sum()
    other_asia_capacity_5 = data_remove.loc[(data_remove['Year'] >= years[j - rolling_capacity]) &
                                            (data_remove['Year'] <= years[j]) &
                                            (data_remove['Continent'] == 'Other Asia'), 'Capacity'].sum()
    # Total Capacity
    total = europe_capacity_5 + east_asia_capacity_5 + north_america_capacity_5 \
            + oceania_capacity_5 + south_america_capacity_5 + other_asia_capacity_5

    # Compute % contribution for each country
    europe_cont = europe_capacity_5/total
    east_asia_cont = east_asia_capacity_5 / total
    nth_america_cont = north_america_capacity_5 / total
    oceania_cont = oceania_capacity_5 / total
    sth_america_cont = south_america_capacity_5 / total
    other_asia_cont = other_asia_capacity_5 / total

    # Make pdf of total contribution
    pdf_contribution = np.array([europe_cont, east_asia_cont, nth_america_cont, oceania_cont, sth_america_cont, other_asia_cont])
    print("Test total probability sums to 1 ", europe_cont + east_asia_cont + nth_america_cont
          + oceania_cont + sth_america_cont + other_asia_cont)

    # Country variance matrix
    country_variance = np.zeros((len(pdf_contribution), len(pdf_contribution)))

    # Loop over all the countries in the pdf
    for x in range(len(pdf_contribution)):
        for y in range(len(pdf_contribution)):
            country_x_density = pdf_contribution[x]
            country_y_density = pdf_contribution[y]
            lats_x = location["Latitude"][x]
            lats_y = location["Latitude"][y]
            longs_x = location["Longitude"][x]
            longs_y = location["Longitude"][y]
            geodesic_distance = haversine(longs_x, lats_x, longs_y, lats_y)

            # Compute country variance
            country_variance[x, y] = geodesic_distance ** 2 * country_x_density * country_y_density

    # Sum above the diagonal
    upper_distance_sum = np.triu(country_variance).sum() - np.trace(country_variance)
    geodesic_variance.append(upper_distance_sum)
    print("Iteration " + str(j) + " / " + str(len(years)))

# Time-varying geodesic variance
plt.plot(geodesic_variance)
plt.xlabel("Time")
plt.ylabel("Geodesic Wasserstein Capacity Variance")
plt.title("Spatial variance")
plt.savefig("Geodesic_green_variance_capacity_variance")
plt.show()
