import matplotlib.pyplot as plt 

import numpy as np

import pandas as pd 

import sklearn.linear_model

# load data
baby_data = pd.read_csv("Data/Popular_Baby_Names.csv")

# keys 
print(baby_data.keys())

# unique birth years in panda frame 
year_of_birth = baby_data['Year of Birth']
print(year_of_birth.unique())

# unique ethnicity
ethnicity = baby_data['Ethnicity']
print(ethnicity.unique())

# unique ethnicity
first_name = baby_data["Child's First Name"]
print(first_name.unique())
print(first_name.unique().size)

# top name in each ethnicity (conditional)
# access group of row/cols using loc 
top_data = baby_data.loc[ baby_data['Rank'] ==1, ['Gender', 'Ethnicity' , "Child's First Name"] ]
# print(top_data)

# Olivia named ranking
top_rank = baby_data.loc[ ( baby_data["Rank"] == 1 ) & (baby_data["Gender"] == "MALE" )  ].drop_duplicates(subset="Count") , ["Year of Birth",'Gender', 'Ethnicity', "Child's First Name"] # [[]] returns a dataframe 

# sort by
top_rank[0].sort_values( by = [ "Year of Birth", "Count"], inplace=True, ascending=False )
print(top_rank[0])

# top_rank = top_rank.to_frame()
# print(type(top_rank) )
# print( top_rank[1] )
