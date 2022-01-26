import joblib
import pandas as pd
import xlsxwriter

joblibmodel = joblib.load("BigFivePersonalityTestModel(2).joblib")

# Predicting the Clusters
pd.options.display.max_columns = 10
predictions = joblibmodel.labels_

my_data = pd.read_excel('https://res.cloudinary.com/prometheusapi/raw/upload/v1643223319/my_personality_test_hyxsvn.xlsx', engine='openpyxl')

my_personality = joblibmodel.predict(my_data)
print('My Personality Cluster: ', my_personality)

# Summing up the my question groups
col_list = list(my_data)
ext = col_list[0:10]
est = col_list[10:20]
agr = col_list[20:30]
csn = col_list[30:40]
opn = col_list[40:50]

my_sums = pd.DataFrame()
my_sums['extroversion'] = my_data[ext].sum(axis=1)/10
my_sums['neurotic'] = my_data[est].sum(axis=1)/10
my_sums['agreeable'] = my_data[agr].sum(axis=1)/10
my_sums['conscientious'] = my_data[csn].sum(axis=1)/10
my_sums['open'] = my_data[opn].sum(axis=1)/10
my_sums['cluster'] = my_personality
print('Sum of my question groups')
print(my_sums)
