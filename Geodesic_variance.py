from scipy.spatial.distance import correlation
import pandas as pd
import numpy as np
from pyemd import emd, emd_with_flow
from math import radians, cos, sin, asin, sqrt

# Read in dataset
data = pd.read_csv("/Users/tassjames/Desktop/carbon_credits_research/hydrogen_research/Hydrogen_data.csv")
data['Capacity'] = pd.to_numeric(data['Capacity'])
data['Year'] = pd.to_numeric(data['Year'])
data_remove = data.dropna()

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
lats_array = np.array(data_remove['Latitude'])
longs_array = np.array(data_remove['Longitude'])
test = len(data_remove['Longitude'])
geographic_distance_matrix = np.zeros((len(data_remove), len(data_remove)))
for i in range(len(lats_array)):
    for j in range(len(longs_array)):
        lats_i = lats_array[i]
        lats_j = lats_array[j]
        longs_i = longs_array[i]
        longs_j = longs_array[j]
        distance = haversine(longs_i, lats_i, longs_j, lats_j)
        geographic_distance_matrix[i, j] = distance

# Generate grid of years
years = np.linspace(2000, 2028, 29)
for i in range(len(years)):
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

        print("Test total probability sums to 1 ", europe_cont + east_asia_cont + nth_america_cont
              + oceania_cont + sth_america_cont + other_asia_cont)
        block = 1

        # # Loop over time series
        # counter = 0
        # geodesic_variance = []
        # for t in range(len(new_cases_burn[0])):  # Looping over time
        #     cases_slice = new_cases_burn[:, t]  # Slice of cases in time
        #     cases_slice_pdf = np.nan_to_num(cases_slice / np.sum(cases_slice))
        #
        #     # Country variance matrix
        #     country_variance = np.zeros((len(cases_slice_pdf), len(cases_slice_pdf)))
        #
        #     # Loop over all the countries in the pdf
        #     for x in range(len(cases_slice_pdf)):
        #         for y in range(len(cases_slice_pdf)):
        #             # # Print state names
        #             # print(names_dc[x])
        #             # print(names_dc[y])
        #             country_x_density = cases_slice_pdf[x]
        #             country_y_density = cases_slice_pdf[y]
        #             lats_x = usa_location_data["Latitude"][x]
        #             lats_y = usa_location_data["Latitude"][y]
        #             longs_x = usa_location_data["Longitude"][x]
        #             longs_y = usa_location_data["Longitude"][y]
        #             geodesic_distance = haversine(longs_x, lats_x, longs_y, lats_y)
        #
        #             # Compute country variance
        #             country_variance[x, y] = geodesic_distance ** 2 * country_x_density * country_y_density