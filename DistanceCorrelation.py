from scipy.spatial.distance import correlation
import pandas as pd
import numpy as np

data = pd.read_csv("/Users/tassjames/Desktop/carbon_credits_research/hydrogen_research/Hydrogen_data.csv")
data['Capacity'] = pd.to_numeric(data['Capacity'])
data['Year'] = pd.to_numeric(data['Year'])
data_remove = data.dropna()

# Generate grid of years
years = np.linspace(2000, 2028, 29)
for i in range(len(years)):

    # 5 year rolling Output
    rolling_capacity = 4 # This is Python indexing
    europe_list = []  # Average Capacity
    for j in range(rolling_capacity, len(years)):
        europe_capacity_5 = data_remove.loc[(data_remove['Year'] >= years[j-rolling_capacity]) &
                                     (data_remove['Year'] <= years[j]) &
                                     (data_remove['Continent'] == 'Europe'), 'Capacity']
        east_asia_capacity_5 = data_remove.loc[(data_remove['Year'] >= years[j - rolling_capacity]) &
                                            (data_remove['Year'] <= years[j]) &
                                            (data_remove['Continent'] == 'East Asia'), 'Capacity']
        north_america_capacity_5 = data_remove.loc[(data_remove['Year'] >= years[j - rolling_capacity]) &
                                               (data_remove['Year'] <= years[j]) &
                                               (data_remove['Continent'] == 'North America'), 'Capacity']
        oceania_capacity_5 = data_remove.loc[(data_remove['Year'] >= years[j - rolling_capacity]) &
                                                   (data_remove['Year'] <= years[j]) &
                                                   (data_remove['Continent'] == 'Oceania'), 'Capacity']
        south_america_capacity_5 = data_remove.loc[(data_remove['Year'] >= years[j - rolling_capacity]) &
                                             (data_remove['Year'] <= years[j]) &
                                             (data_remove['Continent'] == 'South America'), 'Capacity']
        other_asia_capacity_5 = data_remove.loc[(data_remove['Year'] >= years[j - rolling_capacity]) &
                                                   (data_remove['Year'] <= years[j]) &
                                                   (data_remove['Continent'] == 'Other Asia'), 'Capacity']




    # Slice a particular event
    event_m = genders[0][(genders[0]["event"] == events_list_m[i])]
    event_f = genders[1][(genders[1]["event"] == events_list_w[i])]

    # Mean/event/year - men
    means_m = [] # Average Male distance
    for j in range(len(years)):
        mean_year_event = event_m.loc[(event_m['Date'] == years[j]), 'Mark'].mean()
        means_m.append(mean_year_event)


