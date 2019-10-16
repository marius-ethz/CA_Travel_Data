# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:23:39 2019

@author: mschwarz
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

matplotlib.rcdefaults()
#Plot Styles
plt.rcParams.update({'font.size': 12, 'font.family': 'Calibri'})
plt.rc('figure', titlesize=12)
plt.rc('axes', axisbelow=True)
plt.rcParams["figure.figsize"] = (20,10)

#Set working directory
os.chdir("Z:/Public/997 Collaboration/Marius Schwarz/03_VGI in California/00_Marius Schwarz/99_Python Code/02_CA_Travel_Data")  


#%%
########################################################################################
####################   1. Create Initial Dataset 'data'   ##############################
########################################################################################

# In section 1, we create the initial dataset 'data' that is used for the research project 'EVs in California'. To do so, in section 1.1, we read in the two csv files for the NHTS 2017 (i) trip and (ii) houehold data and join these two dataframes. In section 1.2., 

###################      NHTS 2017 trip and household data         #####################

#trip_data_nhts
trip_data_nhts = pd.read_csv("../../05_CA Travel data/0_OriginalData/NHTS2017_DataFiles_Csv_retrieved_29052018/trippub.csv")

trip_data = trip_data_nhts[['HOUSEID','PERSONID','VEHID','TDTRPNUM','STRTTIME','ENDTIME','TRVLCMIN','TRPMILES',\
                      'TRPTRANS','WHYFROM','WHYTO','TDWKND','DRVR_FLG','HHFAMINC','HHSTATE','HTPPOPDN','WTTRDFIN']]


## read household data
hh_data_nhts = pd.read_csv("../../05_CA Travel data/0_OriginalData/NHTS2017_DataFiles_Csv_retrieved_29052018/hhpub.csv")

## join households weights from hh_data to the trip_data
hh_data = hh_data_nhts[['HOUSEID','WTHHFIN','HHVEHCNT']]
trip_data_with_hhweights = trip_data.join(hh_data.set_index('HOUSEID'), on='HOUSEID')
trip_data = trip_data_with_hhweights.copy()

#number of trips
print('nb of trips: ', len(trip_data))
#number of households: Expected = 129,000
print('nb of households:', len(trip_data["HOUSEID"].unique()))

#%%

###################             Pre-process data              #####################

##########   Filtering CA only
trip_data_ca = trip_data.loc[trip_data['HHSTATE'].isin(['CA'])]

#list_of_households_in_NHTS_CA = sorted(list(trip_data_ca['HOUSEID'].unique()))

print('nb of trips CA:', trip_data_ca.shape[0])
print('Percentage of full nhts dataset size', round(100 * trip_data_ca.shape[0] / trip_data.shape[0], 2), '%')
#number of households in CA: Expected = 24,000
#print('NHTS CA households:', len(trip_data_ca['HOUSEID'].unique()))


########## Filtering light-duty vehicles only
## select only light-duty vehicles: 03-car, 04-SUV, 05-Van/Minivan 
trip_data_ca_private_ldv = trip_data.loc[(trip_data['TRPTRANS'].isin([3,4,5])) & (trip_data['HHSTATE'].isin(['CA']))]   
print('nb of trips CA private ldv:', trip_data_ca_private_ldv.shape[0])
print('Percentage of CA trips', round(100 * trip_data_ca_private_ldv.shape[0] / trip_data_ca.shape[0], 2), '%')
print('Percentage of Total trips', round(100 * trip_data_ca_private_ldv.shape[0] / trip_data.shape[0], 2), '%')

#print('CA_ldv households:', len(trip_data_ca_private_ldv["HOUSEID"].unique()))
#data_ca_private_ldv.head(6)


########## Filtering unique vehicle trips only
## select only vehicle trips (DRVR_FLG = 1)
trip_data_ca_private_ldv_vehtrips = trip_data_ca_private_ldv.loc[trip_data['DRVR_FLG'] == 1]
print('Nb private ldv vehtrips CA:', trip_data_ca_private_ldv_vehtrips.shape[0])
print('Percentage of CA trips', round(100 * len(trip_data_ca_private_ldv_vehtrips) / len(trip_data_ca), 2), '%')
print('Percentage of Total trips', round(100 * len(trip_data_ca_private_ldv_vehtrips) / len(trip_data), 2), '%')

#print('trip miles: ', round(sum(trip_data_ca_private_ldv_vehtrips.TRPMILES)))

# Cleen data set
# Variables in Data set: HOUSEID','PERSONID','TDTRPNUM','STRTTIME','ENDTIME','TRVLCMIN','TRPMILES','TRPTRANS','VEHID',WHYFROM','TDWKND','DRVR_FLG','WHYTO','HHFAMINC','HHSTATE','WTTRDFIN','HTPPOPDN'
# Variables in Data set to clean: 'STRTTIME'
# A) STRTTIME - check between 0000 and 2359
# B) ENDTIME  - check between 0000 and 2359
# C) TRVLCMIN - check whether -9 and between 0-1200 minutes
# D) TRPMILES - check whether -9 and 0-9622                    

print('Nb rows cleaning for STRTTIME:', trip_data_ca_private_ldv_vehtrips.loc[(trip_data_ca_private_ldv_vehtrips.STRTTIME < 0)| (trip_data_ca_private_ldv_vehtrips.STRTTIME > 2359)  , 'STRTTIME'].count())
print('Nb rows cleaning for ENDTIME:',  trip_data_ca_private_ldv_vehtrips.loc[(trip_data_ca_private_ldv_vehtrips.ENDTIME < 0)| (trip_data_ca_private_ldv_vehtrips.ENDTIME > 2359)  , 'ENDTIME'].count())
print('Nb rows cleaning for TRPMILES:', trip_data_ca_private_ldv_vehtrips.loc[(trip_data_ca_private_ldv_vehtrips.TRPMILES == -9) | (trip_data_ca_private_ldv_vehtrips.TRPMILES < 0)| (trip_data_ca_private_ldv_vehtrips.TRPMILES > 9622)  , 'TRPMILES'].count())
print('Nb rows cleaning for TRVLCMIN:', trip_data_ca_private_ldv_vehtrips.loc[(trip_data_ca_private_ldv_vehtrips.TRVLCMIN == -9) | (trip_data_ca_private_ldv_vehtrips.TRVLCMIN < 0)| (trip_data_ca_private_ldv_vehtrips.TRVLCMIN > 1200)  , 'TRVLCMIN'].count())

# C) TRVLCMIN
trip_data_ca_private_ldv_vehtrips_clean=trip_data_ca_private_ldv_vehtrips[(trip_data_ca_private_ldv_vehtrips.TRVLCMIN != -9) & (trip_data_ca_private_ldv_vehtrips.TRVLCMIN > 0) & (trip_data_ca_private_ldv_vehtrips.TRVLCMIN < 1200)]

# D) TRPMILES
trip_data_ca_private_ldv_vehtrips_clean=trip_data_ca_private_ldv_vehtrips_clean[(trip_data_ca_private_ldv_vehtrips_clean.TRPMILES != -9) & (trip_data_ca_private_ldv_vehtrips_clean.TRPMILES > 0) & (trip_data_ca_private_ldv_vehtrips_clean.TRPMILES < 9622)]



print('Nb rows cleaning for TRPMILES - after cleaning:', trip_data_ca_private_ldv_vehtrips_clean.loc[(trip_data_ca_private_ldv_vehtrips_clean.TRPMILES == -9) | (trip_data_ca_private_ldv_vehtrips_clean.TRPMILES < 0)| (trip_data_ca_private_ldv_vehtrips_clean.TRPMILES > 9622)  , 'TRPMILES'].count())
print('Nb rows cleaning for TRVLCMIN:', trip_data_ca_private_ldv_vehtrips_clean.loc[(trip_data_ca_private_ldv_vehtrips_clean.TRVLCMIN == -9) | (trip_data_ca_private_ldv_vehtrips_clean.TRVLCMIN < 0)| (trip_data_ca_private_ldv_vehtrips_clean.TRVLCMIN > 1200)  , 'TRVLCMIN'].count())

print('Nb private ldv vehtrips CA clean:', trip_data_ca_private_ldv_vehtrips_clean.shape[0])
print('Percentage of CA trips', round(100 * len(trip_data_ca_private_ldv_vehtrips_clean) / len(trip_data_ca), 2), '%')
print('Percentage of Total trips', round(100 * len(trip_data_ca_private_ldv_vehtrips_clean) / len(trip_data), 2), '%')

#%%

# Add unique identifiers for vehicles
trip_data_ca_private_ldv_vehtrips_clean['HOUSEID'] = trip_data_ca_private_ldv_vehtrips_clean['HOUSEID'].astype(str)
trip_data_ca_private_ldv_vehtrips_clean['VEHID'] = trip_data_ca_private_ldv_vehtrips_clean['VEHID'].astype(str)

trip_data_ca_private_ldv_vehtrips_clean['HOUSEVEHID'] = trip_data_ca_private_ldv_vehtrips_clean['HOUSEID'] + trip_data_ca_private_ldv_vehtrips_clean['VEHID']

trip_data_ca_private_ldv_vehtrips_clean = trip_data_ca_private_ldv_vehtrips_clean[['HOUSEVEHID','TDTRPNUM','STRTTIME','ENDTIME','TRVLCMIN','TRPMILES',\
                      'TRPTRANS','WHYFROM','WHYTO','TDWKND', 'HHFAMINC', 'HTPPOPDN','WTTRDFIN','WTHHFIN','HHVEHCNT']]

#%%

# Show final dataframe 'data'

data = trip_data_ca_private_ldv_vehtrips_clean.copy().reset_index(drop=True)

nb_of_trips = len(data)
nb_of_vehicles = len(data['HOUSEVEHID'].unique())

nb_of_trips_CA = len(trip_data_ca)
nb_of_trips_total = len(trip_data)


print('nb_of_trips_total:', nb_of_trips_total)
print('nb_of_trips_CA:', nb_of_trips_CA)
print('nb_of_trips in preprocessed data: ', nb_of_trips)

print('percentage_of_trips_remaining_from_CA_trips: ',
      round(100*nb_of_trips / nb_of_trips_CA, 2), '%')
print('percentage_of_trips_remaining_from_total_number_of_trips: ',
      round(100*nb_of_trips / nb_of_trips_total, 2), '%')

print('===================================')
print('nb_of_vehicles in preprocessed data: ', nb_of_vehicles)

#%%
########################################################################################
####################   2. Detailed overview of vehicle driving patterns     ############
########################################################################################

# In this chapter, we develop a more detailed view on the vehicle driving patterns compared to the previous version. We aim to understand in detail where which car is during each hour or 15 minutes timeslot. We thus not only look at the start time of the first trip and the end time of the last trip, but also at the trips inbetween. 

####################   Create new dataframe 'data_all_trips'     ############

data_trip_nb = data.copy().sort_values(['HOUSEVEHID','STRTTIME','ENDTIME']).reset_index(drop=True)

#We will later use the "WHYTO" and 'WHYFROM" Columns to define the location of the vehicle. To use this column, we reassign the values in this column as follow:
#
#    '-9'=notascertain ---- '00' No allocation
#    '8'=I don't know ---- '00' No allocation
#    '7'=I prefer not to answer ---- '00' No allocation
#    '1'=Regular home activities (chores, sleep) ---- '01' Home
#    '2'=Work from home (paid) ---- '01' Home
#    '3'=Work ---- '02' Work
#    '4'=Work-related meeting / trip ---- '02' Work
#    '5'=Volunteer activities (not paid) ---- '03' Public
#    '6'=Drop off /pick up someone ---- '03' Public
#    '7'=Change type of transportation ---- '03' Public
#    '8'=Attend school as a student ---- '03' Public
#    '9'=Attend child care ---- '03' Public
#    '10'=Attend adult care ---- '03' Public
#    '11'=Buy goods (groceries, clothes, appliances, gas) ---- '03' Public
#    '12'=Buy services (dry cleaners, banking, service a car, pet care) ---- '03' Public
#    '13'=Buy meals (go out for a meal, snack, carry-out) ---- '03' Public
#    '14'=Other general errands (post office, library) ---- '03' Public
#    '15'=Recreational activities (visit parks, movies, bars, museums) ---- '03' Public
#    '16'=Exercise (go for a jog, walk, walk the dog, go to the gym) ---- '03' Public
#    '17'=Visit friends or relatives ---- '01' Home
#    '18'=Health care visit (medical, dental, therapy) ---- '03' Public
#    '19'=Religious or other community activities ---- '03' Public
#    '97'=Something else ---- '00' No allocation

# Reassigning values in "WHYTO" column
data_trip_nb.at[data_trip_nb.WHYTO == -9, 'WHYTO'] = 'NA'
data_trip_nb.at[data_trip_nb.WHYTO == -8, 'WHYTO'] = 'NA'
data_trip_nb.at[data_trip_nb.WHYTO == -7, 'WHYTO'] = 'NA'
data_trip_nb.at[data_trip_nb.WHYTO == 1, 'WHYTO'] = 'Home'
data_trip_nb.at[data_trip_nb.WHYTO == 2, 'WHYTO'] = 'Home'
data_trip_nb.at[data_trip_nb.WHYTO == 3, 'WHYTO'] = 'Work'
data_trip_nb.at[data_trip_nb.WHYTO == 4, 'WHYTO'] = 'Work'
data_trip_nb.at[data_trip_nb.WHYTO == 5, 'WHYTO'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYTO == 6, 'WHYTO'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYTO == 7, 'WHYTO'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYTO == 8, 'WHYTO'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYTO == 9, 'WHYTO'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYTO == 10, 'WHYTO'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYTO == 11, 'WHYTO'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYTO == 12, 'WHYTO'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYTO == 13, 'WHYTO'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYTO == 14, 'WHYTO'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYTO == 15, 'WHYTO'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYTO == 16, 'WHYTO'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYTO == 17, 'WHYTO'] = 'Home'
data_trip_nb.at[data_trip_nb.WHYTO == 18, 'WHYTO'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYTO == 19, 'WHYTO'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYTO == 97, 'WHYTO'] = 'NA'

data_trip_nb.at[data_trip_nb.WHYFROM == -9, 'WHYFROM'] = 'NA'
data_trip_nb.at[data_trip_nb.WHYFROM == -8, 'WHYFROM'] = 'NA'
data_trip_nb.at[data_trip_nb.WHYFROM == -7, 'WHYFROM'] = 'NA'
data_trip_nb.at[data_trip_nb.WHYFROM == 1, 'WHYFROM'] = 'Home'
data_trip_nb.at[data_trip_nb.WHYFROM == 2, 'WHYFROM'] = 'Home'
data_trip_nb.at[data_trip_nb.WHYFROM == 3, 'WHYFROM'] = 'Work'
data_trip_nb.at[data_trip_nb.WHYFROM == 4, 'WHYFROM'] = 'Work'
data_trip_nb.at[data_trip_nb.WHYFROM == 5, 'WHYFROM'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYFROM == 6, 'WHYFROM'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYFROM == 7, 'WHYFROM'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYFROM == 8, 'WHYFROM'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYFROM == 9, 'WHYFROM'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYFROM == 10, 'WHYFROM'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYFROM == 11, 'WHYFROM'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYFROM == 12, 'WHYFROM'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYFROM == 13, 'WHYFROM'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYFROM == 14, 'WHYFROM'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYFROM == 15, 'WHYFROM'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYFROM == 16, 'WHYFROM'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYFROM == 17, 'WHYFROM'] = 'Home'
data_trip_nb.at[data_trip_nb.WHYFROM == 18, 'WHYFROM'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYFROM == 19, 'WHYFROM'] = 'Public'
data_trip_nb.at[data_trip_nb.WHYFROM == 97, 'WHYFROM'] = 'NA'

#%%
# defining the functions used to create the trip_cat column
def get_trip_nb(previous_row_houseid, current_row_houseid, counter):
         
        if current_row_houseid != previous_row_houseid:   # first trip of the day: set trip_cat = 1
            return(1)
        else:                                                   # middle trips: set trip_cat = 0
            return(counter + 1) 

#%%
            
# defining first trip and last trip as the first trip of the vehicle of the household and the last trip of this same vehicle.
def create_trip_nb_list(df):
        
    # initialise the list that will be passed into the new trip_cat dataframe column
    trip_nb_list = []
    counter = 1
    
    # get the trip_cat value for each row index and append it to trip_cat_list
    for index,row in df.iterrows():
        
        row_index = index
        
        if row_index == 0:  #initialise first trip of first vehicle
            trip_nb_value = 1
        
        else:
            #start_time = time.time()
            
            previous_row_houseid = df.iloc[row_index - 1]['HOUSEVEHID']
            current_row_houseid = df.iloc[row_index]['HOUSEVEHID']
                         
            trip_nb_value = get_trip_nb(previous_row_houseid, current_row_houseid, \
                                                            counter)
            counter = trip_nb_value
            
        # append this value to trip_cat_list
        trip_nb_list.append(trip_nb_value)
        
    # return the full list of trip_cat values   
    return(trip_nb_list)

#%%

# create the trip_cat list (calculation takes some time, around 1-2 minutes)
trip_nb_list = create_trip_nb_list(data_trip_nb)

#%%

# insert the trip_cat list as a column in the dataframe
data_trip_nb.insert(3, 'TRIPNB', trip_nb_list)

#%%

# Seperate in x dictionaries for each trip, whereby x is the max number of trips for one vehicles
def create_trip_nb_dictionaries(df):
    
    #Set up for loop for x between 1 and nb_new_d which we define below
    for x in range(1,nb_new_d):
        
        # We create x = nb_new_d dictionaries for each x-th trip of households
        d[x] = df[df.TRIPNB == x][['HOUSEVEHID','TRIPNB','STRTTIME','ENDTIME',\
                                         'TRPMILES','TRPTRANS', 'WHYFROM', 'WHYTO']]

#%%

d = {} # Initializing dictionaries
nb_new_d = data_trip_nb['TRIPNB'].max() # Max trips per household

create_trip_nb_dictionaries(data_trip_nb) # Call function to create dictionaries

#%%

#Create new dataframe with all trips 'data_all_trips', using the above created dictionaries
def create_data_all_trips(df):
    #creating local dataframwork that we return in the end of the function
    df_new_1 = df.copy()
    
    # Add dictionaries to the local dataframework in a for loop
    for x in range(1,nb_new_d):
        df_new_2 = pd.DataFrame(d[x]) #Transforming the dictionary in a dataframe
        df_new_1 = df_new_1.join(df_new_2.set_index('HOUSEVEHID'), on = 'HOUSEVEHID', \
                                 rsuffix = '_' + str(x), \
                                 lsuffix = '_' + str(x - 1)) 
        
    return(df_new_1)
    
#%%
    
# Preparing the dataframe 'data_all_trips'
# Keep only columns that are used for all trips
data_all_trips = data_trip_nb[['HOUSEVEHID', 'TDWKND', 'HHFAMINC', 'HTPPOPDN', 'WTTRDFIN', \
                       'WTHHFIN', 'HHVEHCNT']]

data_all_trips = data_all_trips.groupby('HOUSEVEHID').mean().reset_index()


#%%

# Call function to add dictionaries to new dataframe
data_all_trips = create_data_all_trips(data_all_trips)

#%%

##########################     New DataFrame 'data_all_trips_decihour'   ############################

# copy paste to create new dataframe
data_all_trips_decihour = data_all_trips.copy()

#%%
# define the function to transform one HHMM input into decimal hours (will be used inside the dataframe with applymap function)
def from_hhmm_to_decimal(input):
   
    if math.isnan(input):
        decimal_hour = input
        
        
    else:
        input = int(input)
        mystring = str(input)
        m = float(mystring[-2:])
        
        if mystring[:-2] == '':
            h = float(0)
        
        else:
            h = float(mystring[:-2])
        
        decimal_hour = float(h) + float(m)/60
    
    return(decimal_hour)
    
#%%

# convert the columns with start times to decimals
data_all_trips_decihour.loc[:, data_all_trips_decihour.columns.str.startswith('STRTTIME')] \
= data_all_trips_decihour.loc[:, data_all_trips_decihour.columns.str.startswith('STRTTIME')\
                             ].applymap(from_hhmm_to_decimal)

# convert the columns with end times to decimals
data_all_trips_decihour.loc[:, data_all_trips_decihour.columns.str.startswith('ENDTIME')] \
= data_all_trips_decihour.loc[:, data_all_trips_decihour.columns.str.startswith('ENDTIME')\
                             ].applymap(from_hhmm_to_decimal)

#%%
##########################     Add 'hour-columns' to dataset   ############################

# Setup new dataframe including the hourly profile
data_all_trips_decihour_profile = data_all_trips_decihour.copy()

# Add the hourly profile
# The NHTS Travel Survey perceives a travel day as from 4am to 4am. To better process the data we create first columns from 4 to 28, we change that later again 
for x in range(4,28):
    data_all_trips_decihour_profile[str(x) + 'h'] = 'Home'

#%%
    
def add_profile_per_vehicle(df):
    
    df_return = df.copy()
    
    for index,row in df.iterrows():
        
        row_index = index
                
        for column_index in range(1,33): #The max number of trips of onle vehicle is 33
            
            if math.isnan(df_return.at[row_index, 'STRTTIME_' + str(column_index)]):
                pass
            
            else:
                start_time = df_return.at[row_index, 'STRTTIME_' + str(column_index)]
                start_time = int(round(start_time))
                end_time   = df_return.at[row_index, 'ENDTIME_' + str(column_index)]
                end_time   = int(round(end_time))
                where_to   = df_return.at[row_index, 'WHYTO_' + str(column_index)]
                where_from = df_return.at[row_index, 'WHYFROM_' + str(column_index)]
                
                #Account for that not all households started there first trip from home
                if column_index == 1:
                    for before_first_trip in range(4,start_time):
                        df_return.at[row_index, str(before_first_trip) + 'h'] =  where_from                   
                
                # We have to account for that the travel day is from 4am to 4am 
                if start_time < 4: 
                    start_time = start_time + 24
                
                if end_time <= 4:
                    end_time = end_time + 24
                    
                for driving in range(start_time,end_time):
                    df_return.at[row_index, str(driving) + 'h'] = 'Driving'
                
                for rest_of_day in range(end_time,28):
                    df_return.at[row_index, str(rest_of_day) + 'h'] = where_to
    
    return(df_return)

#%%
    
# call function 'add_profile_per_vehicle', function takesaround 20 sec 
data_all_trips_decihour_profile = add_profile_per_vehicle(data_all_trips_decihour_profile)

#Change column headers for hours
data_all_trips_decihour_profile = data_all_trips_decihour_profile.rename(columns={"24h": "0h", "25h": "1h",\
                                                "26h": "2h", "27h": "3h",})
    
#%%
##################   Create list for home, work, and public     ########################
    
def create_travel_patterns_all_trips(df, condition):
    my_list = np.array([0]*24)
    
    for index,row in df.iterrows():
        row_index = index
        
        for column_index in range(0,24):
            if df.at[row_index, str(column_index) + 'h'] == condition:
                my_list[column_index] = my_list[column_index] + 1    
            
    return(my_list)   

#%%
condition_none = 'NA'
condition_home = 'Home'
condition_work = 'Work'
condition_public = 'Public'
condition_driving = 'Driving'


travel_pattern_none = create_travel_patterns_all_trips(data_all_trips_decihour_profile,\
                                                       condition_none).copy()

travel_pattern_home = create_travel_patterns_all_trips(data_all_trips_decihour_profile,\
                                                       condition_home).copy()

travel_pattern_work = create_travel_patterns_all_trips(data_all_trips_decihour_profile,\
                                                       condition_work).copy()

travel_pattern_public = create_travel_patterns_all_trips(data_all_trips_decihour_profile,\
                                                       condition_public).copy()

travel_pattern_driving = create_travel_patterns_all_trips(data_all_trips_decihour_profile,\
                                                       condition_driving).copy()

#%%
  
#Create percentage
travel_pattern_driving_per = travel_pattern_driving / len(data_all_trips_decihour_profile['1h'])
travel_pattern_home_per = travel_pattern_home / len(data_all_trips_decihour_profile['1h'])
travel_pattern_work_per = travel_pattern_work / len(data_all_trips_decihour_profile['1h'])
travel_pattern_public_per = travel_pattern_public / len(data_all_trips_decihour_profile['1h'])
travel_pattern_none_per = travel_pattern_none / len(data_all_trips_decihour_profile['1h'])

#%%
##############################   PLOT 2: Travel Patterns    #########################################

# Create Plot with #vehicles
plt.figure(figsize=(10,6))

#x and y-axis
x=range(0,24)
plt.title('Overview Driving Pattern', fontsize = 14, y = 1.05) #dpi = 300
 
cmap = matplotlib.cm.get_cmap('Greys')
font_title = {'size':'15', 'color':'black', 'weight':'bold',
              'verticalalignment':'bottom'}
font_axis = {'size':'12', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}


c = [cmap(0.2), cmap(0.4), cmap(0.6), cmap(0.9)]

plt.stackplot(x, travel_pattern_driving, travel_pattern_home, travel_pattern_work, 
              travel_pattern_public, 
              labels=['Driving','Home', 'Work', 'Public'], 
              colors = c,
              alpha = 0.7)

# Axes
plt.ylim((0,25500))
plt.xlim((0,23))
plt.grid(axis='y')
plt.xlabel('[h]')
plt.ylabel('[# Vehicles]')
        
# Legend
plt.legend(loc='lower right', frameon=True, fancybox=False)
plt.savefig('overview_travel_pattern.jpeg', dpi = 300, transparent = False, bbox_inches="tight")

plt.show()


#%%

# Create Plot with #vehicles
plt.figure(figsize=(10,6))

#x and y-axis
x=range(0,24)
plt.title('Overview Driving Pattern', fontsize = 14, y = 1.05) #dpi = 300
 
cmap = matplotlib.cm.get_cmap('Greys')
font_title = {'size':'15', 'color':'black', 'weight':'bold',
              'verticalalignment':'bottom'}
font_axis = {'size':'12', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}


c = [cmap(0.2), cmap(0.4), cmap(0.6), cmap(0.9)]

plt.stackplot(x, travel_pattern_driving_per, travel_pattern_home_per, travel_pattern_work_per, 
              travel_pattern_public_per,
              labels=['Driving','Home', 'Work', 'Public'], 
              colors = c,
              alpha = 0.7)

# Axes
plt.ylim((0,1))
plt.xlim((0,23))
plt.grid(axis='y')
plt.xlabel('[h]')
plt.ylabel('[# Vehicles]')
        
# Legend
plt.legend(loc='lower right', frameon=True, fancybox=False)
plt.savefig('overview_travel_pattern_percent.jpeg', dpi = 300, transparent = False, bbox_inches="tight")

plt.show()

#%%

#!!! The following plot can only be created if in addition the python code "NHTS_2017_TripDate_Preprocessing" has been run on the same kernel. 
fig = plt.figure(constrained_layout=True, figsize=(10,8), dpi=300)
gs = fig.add_gridspec(2, 3)
#x and y-axis

cmap = matplotlib.cm.get_cmap('Greys')
font_title = {'size':'15', 'color':'black', 'weight':'bold',
              'verticalalignment':'bottom'}
font_axis = {'size':'12', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
c = [cmap(0.2), cmap(0.4), cmap(0.6), cmap(0.9)]

# Content
fig_ax1 = fig.add_subplot(gs[0, :])
x=range(0,24) 
fig_ax1.stackplot(x, travel_pattern_driving_per, travel_pattern_home_per, travel_pattern_work_per, 
              travel_pattern_public_per,
              labels=['Driving','Home', 'Work', 'Public'], 
              colors = c,
              alpha = 0.7)

fig_ax2 = fig.add_subplot(gs[1, :-1])
x = np.linspace (0, 200, 200)
fig_ax2.plot(x, appr_start_time, '-y', label='Normal distr, start time', color='red', alpha=0.6)
fig_ax2.plot(x, appr_end_time, '-y', label='Normal distr, end time', color='blue', alpha=0.6)
fig_ax2.hist(data_tripcat_12_decihour['STRTTIME_first_trip'], bins = 100, color='grey', edgecolor='black', alpha=0.6, density=True, label='Start time of first trip')
fig_ax2.hist(data_tripcat_12_decihour['ENDTIME_last_trip'], bins = 100, color='black', edgecolor='black', alpha=0.6, density=True, label='End time of last trip')

fig_ax3 = fig.add_subplot(gs[1:, -1])
x = np.linspace (0, 400, 400) 
y1 = stats.gamma.pdf(x, a=alpha_daily_miles, scale=scale_daily_miles)
fig_ax3.hist(data_housevehid['TRPMILES'], bins = 100, color='black', edgecolor='black', alpha=0.6, label='Data', density=True)
fig_ax3.plot(x, y1, "y-", label='Gamma distr', color='red', alpha=0.6) 

# Axes
fig_ax1.set_title('a.', **font_title)
fig_ax1.title.set_position([-0.065, 1.1])
fig_ax1.yaxis.grid(True)
fig_ax1.set_xlim([0,23])
fig_ax1.set_ylim([0,1])
fig_ax1.set_xlabel('[h]', labelpad=10)
fig_ax1.set_ylabel('[# Vehicles]', labelpad=10)
fig_ax1.legend(loc='lower right', frameon=True, fancybox=False)

fig_ax2.set_title('b.', **font_title)
fig_ax2.title.set_position([-0.1, 1.1])
fig_ax2.yaxis.grid(True)
fig_ax2.set_xlim([0,24])
fig_ax2.set_ylim([0,0.3])
fig_ax2.set_xlabel('[h]', labelpad=10)
fig_ax2.set_ylabel('[Share of Vehicles]', labelpad=10)
fig_ax2.legend(loc='upper right', frameon=True, fancybox=False)
      
fig_ax3.set_title('c.', **font_title)
fig_ax3.title.set_position([-0.2, 1.1])
fig_ax3.yaxis.grid(True)
fig_ax3.set_xlim([0,400])
fig_ax3.set_ylim([0,0.03])
fig_ax3.set_xlabel('[# Daily Miles per Vehicle]', labelpad=10)
fig_ax3.set_ylabel('[Share of Vehicles]', labelpad=10)
fig_ax3.legend(loc='upper right', frameon=True, fancybox=False)  
        
        
# Plotting
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.savefig('Overview Travel Pattern Plots.jpeg', dpi = 300, transparent = False, bbox_inches="tight")

plt.show()









