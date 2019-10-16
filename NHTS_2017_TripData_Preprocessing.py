# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:32:53 2019

@author: mschwarz
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import csv
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

# In section 1, we create the initial dataset 'data' that is used for the research project 'EVs in California'. 
# To do so, in section 1.1, we read in the two csv files for the NHTS 2017 (i) trip and (ii) houehold data and join these two dataframes. 
# In section 1.2., 


##################    NHTS 2017 trip and household data     ###########################

# Read in trip csv file 
#trip_data_nhts
trip_data_nhts = pd.read_csv("../../05_CA Travel data/0_OriginalData/NHTS2017_DataFiles_Csv_retrieved_29052018/trippub.csv")

# Keep only columns that we also use later
trip_data = trip_data_nhts[['HOUSEID','PERSONID','VEHID','TDTRPNUM','STRTTIME','ENDTIME','TRVLCMIN','TRPMILES',\
                      'TRPTRANS','WHYFROM','WHYTO','TDWKND','DRVR_FLG','HHFAMINC','HHSTATE','HTPPOPDN','WTTRDFIN']]

# read household data
hh_data_nhts = pd.read_csv("../../05_CA Travel data/0_OriginalData/NHTS2017_DataFiles_Csv_retrieved_29052018/hhpub.csv")

# join households weights from hh_data to the trip_data
hh_data = hh_data_nhts[['HOUSEID','WTHHFIN','HHVEHCNT']]
trip_data_with_hhweights = trip_data.join(hh_data.set_index('HOUSEID'), on='HOUSEID')
trip_data = trip_data_with_hhweights.copy()

#number of trips
print('nb of trips: ', len(trip_data))
#number of households: Expected = 129,000
print('nb of households:', len(trip_data["HOUSEID"].unique()))

#%%

##################    Pre-process data     ###########################

###### Filtering California only
trip_data_ca = trip_data.loc[trip_data['HHSTATE'].isin(['CA'])]
print('nb of trips CA:', trip_data_ca.shape[0])
print('Percentage of full nhts dataset size', round(100 * trip_data_ca.shape[0] / trip_data.shape[0], 2), '%')
#number of households in CA: Expected = 24,000
#print('NHTS CA households:', len(trip_data_ca['HOUSEID'].unique()))


###### Filtering light-duty vehicles only
## select only light-duty vehicles: 03-car, 04-SUV, 05-Van/Minivan 
## Light-duty-vehicles: TRPTRANS = 3,4,5
trip_data_ca_private_ldv = trip_data.loc[(trip_data['TRPTRANS'].isin([3,4,5])) & (trip_data['HHSTATE'].isin(['CA']))]   
print('nb of trips CA private ldv:', trip_data_ca_private_ldv.shape[0])
print('Percentage of CA trips', round(100 * trip_data_ca_private_ldv.shape[0] / trip_data_ca.shape[0], 2), '%')
print('Percentage of Total trips', round(100 * trip_data_ca_private_ldv.shape[0] / trip_data.shape[0], 2), '%')


##### Filtering unique vehicle trips only
## select only vehicle trips (DRVR_FLG = 1)
trip_data_ca_private_ldv_vehtrips = trip_data_ca_private_ldv.loc[trip_data['DRVR_FLG'] == 1]
print('Nb private ldv vehtrips CA:', trip_data_ca_private_ldv_vehtrips.shape[0])
print('Percentage of CA trips', round(100 * len(trip_data_ca_private_ldv_vehtrips) / len(trip_data_ca), 2), '%')
print('Percentage of Total trips', round(100 * len(trip_data_ca_private_ldv_vehtrips) / len(trip_data), 2), '%')

#%%

###### Clean Data set

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

######## Add unique identifier for vehicles
trip_data_ca_private_ldv_vehtrips_clean['HOUSEID'] = trip_data_ca_private_ldv_vehtrips_clean['HOUSEID'].astype(str)
trip_data_ca_private_ldv_vehtrips_clean['VEHID'] = trip_data_ca_private_ldv_vehtrips_clean['VEHID'].astype(str)

trip_data_ca_private_ldv_vehtrips_clean['HOUSEVEHID'] = trip_data_ca_private_ldv_vehtrips_clean['HOUSEID'] + trip_data_ca_private_ldv_vehtrips_clean['VEHID']

trip_data_ca_private_ldv_vehtrips_clean = trip_data_ca_private_ldv_vehtrips_clean[['HOUSEVEHID','TDTRPNUM','STRTTIME','ENDTIME','TRVLCMIN','TRPMILES',\
                      'TRPTRANS','WHYFROM','WHYTO','TDWKND', 'HHFAMINC', 'HTPPOPDN','WTTRDFIN','WTHHFIN','HHVEHCNT']]

#%%

############ Show final dataframe 'data'

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

#%%
########################################################################################
####################  Getting statistics for NetLogo      ##############################
########################################################################################

#Which data do we collect here?
#- Average number of daily trips per vehicle
#
#- Mean start time
#- Mean end time
#- Mean daily miles
#- Standard Deviation of 'start time'
#- Standard Deviation of 'end time'
#- Standard Deviation of 'mean daily miles'

#########################   Daily trips per vehicle    ################################

# Dataframe : 'data'
## average number of trips per household-vehicle:

average_nb_trips_per_vehicle = data['WTTRDFIN'].sum()/365 / np.sum((data['WTHHFIN'] * data['HHVEHCNT']).unique())

print('===================================')
print('======= OUTPUTS FOR NETLOGO =======')
print('===================================')
print()
print('average_nb_trips_per_vehicle: '         , average_nb_trips_per_vehicle)


#%%
#########################   Daily miles per vehicle    ################################

# Dataframe: 'data_housevehid'
# To calculate the daily miles per vehicle we create a new dataframe 'data_housevehid'. We aggregate the values when we group the rows by the column 'housevehid'

#############  Create new dataframe 'data_housevehid'

# Create new dataframe and only keep columns that are needed for further calculation
data_housevehid = data[['HOUSEVEHID','TRVLCMIN','TRPMILES','WTTRDFIN']]
 
# Aggregate data when grouping by housevehid
data_housevehid = data_housevehid.groupby('HOUSEVEHID').agg({'TRPMILES': sum, 
                                                             'TRVLCMIN': sum,
                                                             'WTTRDFIN': 'mean'}) 

#%%
# histograms of daily miles to see distribution of data
plt.figure(figsize=(5,5))

plt.title('Overview Daily Miles per Vehicle', fontsize = 14, y = 1) #dpi = 300

plt.hist(data_housevehid['TRPMILES'], bins = 100, color='grey', edgecolor='black', alpha=0.6)

# Axes
plt.xlim((0,400))
#plt.xlim((0,23))
plt.grid(axis='y')
plt.xlabel('# Daily Miles per Vehicle')
plt.ylabel('# Vehicles')

           
plt.savefig('Overview Daily Miles per Vehicle.jpeg', dpi = 300, transparent = False, bbox_inches="tight")
plt.show()

#%%
#The daily trip miles of household-vehicles [TRPMILES] follow a distribution that we can approximate with a Gamma-distribution, with the two distinctive parameters: alpha = mean^2/variance ; scale = variance/mean

## Printing final statistics used for the NetLogo model

#Calculate mean
mean_daily_miles = (data_housevehid['TRPMILES'] * data_housevehid['WTTRDFIN'] / 365).sum() \
                    / (data_housevehid['WTTRDFIN'] / 365).sum()

#Calculate standard deviation
std_daily_miles = ( \
                               (len(data_housevehid['TRPMILES']) * ((((data_housevehid['TRPMILES'] - mean_daily_miles)**2) * data_housevehid['WTTRDFIN'] / 365).sum())) \
                               / ((len(data_housevehid['TRPMILES']) - 1) * (data_housevehid['WTTRDFIN'] / 365).sum()) \
                              )**(1/2)

#Calculate alpha and lambda for Gamma Distribution
alpha_daily_miles = (mean_daily_miles**2) / (std_daily_miles**2)
scale_daily_miles = (std_daily_miles**2)  / mean_daily_miles 

print('===================================')
print('======= OUTPUTS FOR NETLOGO =======')
print('===================================')
print()

print('alpha_daily_miles', alpha_daily_miles)
print('scale_daily_miles', scale_daily_miles)

#%%

#Create Data
x = np.linspace (0, 400, 400) 
y1 = stats.gamma.pdf(x, a=alpha_daily_miles, scale=scale_daily_miles)

#%%

# Comparing approximation with gamma distribution and real data

plt.figure(figsize=(5,5))

plt.title('Comparing data and approximation with gamma distribution', fontsize = 14, y = 1.05) #dpi = 300

plt.hist(data_housevehid['TRPMILES'], bins = 100, color='grey', edgecolor='black', alpha=0.6, label='Data', density=True)
plt.plot(x, y1, "y-", label='Gamma distribution', color='red', alpha=0.6) 

# Axes
plt.xlim((0,400))
plt.ylim((0,0.03))
plt.grid(axis='y')
plt.xlabel('# Daily Miles per Vehicle')
plt.ylabel('Share of Vehicles')
           

plt.legend(loc='upper right')


plt.savefig('Comparing data and approximation with gamma distribution.jpeg', dpi = 300, transparent = False, bbox_inches="tight")
plt.show()

#%%

#############################     First trip and last trip statistics   #####################################
#- dataframe: 'data_tripcat'
#- dataframe: 'data_tripcat_12'

########### Create new dataframe 'data_tripcat'

# Create trip_cat variable: identify first trip and last trip of each household
# sort trips by chronological order (just a check, should be already ordered)
data_tripcat = data.copy().sort_values(['HOUSEVEHID','TDTRPNUM']).reset_index(drop=True)

# defining the functions used to create the trip_cat column
def get_trip_category(previous_row_houseid, current_row_houseid, next_row_houseid):
         
        if current_row_houseid != previous_row_houseid:   # first trip of the day: set trip_cat = 1
            return(1)
        elif current_row_houseid != next_row_houseid:     # last trip of the day: set trip_cat = 2
            return(2)
        else:                                                   # middle trips: set trip_cat = 0
            return(0) 
        
def create_trip_category_list(df):
    
# defining first trip and last trip as the first trip of the vehicle of the household and the last trip of this same vehicle.

    # initialise the list that will be passed into the new trip_cat dataframe column
    trip_cat_list = []
    
    # get the trip_cat value for each row index and append it to trip_cat_list
    for index,row in df.iterrows():
        
        row_index = index
        ## uncomment the following line to follow the running of the code (printing the line being processed)
        #print('new_row: index ', row_index)
        
        if row_index == 0:  #initialise first trip of first vehicle
            trip_cat_value = 1
        if row_index == df.index[-1]:  #initialiuse last trip of last vehicle
            trip_cat_value = 2
        
        else:
            #start_time = time.time()
            
            previous_row_houseid = df.iloc[row_index - 1]['HOUSEVEHID']
            current_row_houseid = df.iloc[row_index]['HOUSEVEHID']
            next_row_houseid = df.iloc[row_index + 1]['HOUSEVEHID']
            
            #print("--- %s seconds step_1 ---" % (time.time() - start_time))
            
            trip_cat_value = get_trip_category(previous_row_houseid, current_row_houseid, \
                                                            next_row_houseid)
            
            #print("--- %s seconds step_3 ---" % (time.time() - start_time))
    
        # append this value to trip_cat_list
        trip_cat_list.append(trip_cat_value)
        
    # return the full list of trip_cat values
    #print(trip_cat_list)   
    return(trip_cat_list)
    
#%%
    
# create the trip_cat list (calculation takes some time, around 1-2 minutes)
trip_cat_list = create_trip_category_list(data_tripcat)

# insert the trip_cat list as a column in the dataframe
data_tripcat.insert(3, 'trip_cat', trip_cat_list)

#%%
#############   Create new dataframe 'data_tripcat_12'

# Dataframe 'data_tripcat_12' includes only first trip and last trip start and end times (used for first and last trips statistics, and preparing for PLOT 1)

# seperate in 2 dataframes for first trips and last trips
data_tripcat_1 = data_tripcat[data_tripcat.trip_cat == 1][['HOUSEVEHID','WTTRDFIN','STRTTIME','ENDTIME','TRPMILES',\
                                                          'WHYFROM', 'WHYTO']]

data_tripcat_2 = data_tripcat[data_tripcat.trip_cat == 2][['HOUSEVEHID','STRTTIME','ENDTIME','TRPMILES',\
                                                          'WHYFROM', 'WHYTO']]

data_tripcat_0 = data_tripcat[data_tripcat.trip_cat == 0][['HOUSEVEHID','STRTTIME','ENDTIME','TRPMILES',\
                                                          'WHYFROM', 'WHYTO']]

print(len(data_tripcat_1))
print(len(data_tripcat_2))
print(len(data_tripcat_0))

# join dataframe to get first trip and last trip on same line for each household
data_tripcat_12 = data_tripcat_1.join(data_tripcat_2.set_index('HOUSEVEHID'), on='HOUSEVEHID', 
                                                                         lsuffix='_first_trip',
                                                                         rsuffix='_last_trip')

print(len(data_tripcat_12))

#%%

# No Last Trip

# Investigate why no last trip for some households: Different length of databases with trip_cat = 1 and trip_cat = 2. Some household only have only 1 trip per day. 
# Consider it as a loop trip home. Consider the first trip as last trip also

#Is the number of looptrips relevant for the overall statistics?

print('households_with_only_one_trip: ', len(data_tripcat_12[data_tripcat_12.STRTTIME_last_trip.isna()]))
print('total_households: ', len(data_tripcat_12[data_tripcat_12.STRTTIME_last_trip.isna()]))
print('%share: ', round(100 * len(data_tripcat_12[data_tripcat_12.STRTTIME_last_trip.isna()])/len(data_tripcat_12),2), '%')
print()

print('daily miles of households with only 1 trip: ', data_tripcat_12[data_tripcat_12.STRTTIME_last_trip.isna()].TRPMILES_first_trip.sum())
print('total daily miles: ', data.TRPMILES.sum())
print('%share: ', round(100 * data_tripcat_12[data_tripcat_12.STRTTIME_last_trip.isna()].TRPMILES_first_trip.sum() / data.TRPMILES.sum(),2), '%')

#The Households with only 1 trip represent 2.72% of the total households and account for 4.16% of the total miles. They therefore drive more than the other households. 
#> It is not appropriate to delete them since 4% of miles is a non-negligeable amount.
#> Therefore for these households with 1 trip only, we count their first trip as last trip too (their only trip is a loop trip home)

data_tripcat_12['STRTTIME_last_trip'] = data_tripcat_12['STRTTIME_last_trip'].fillna(data_tripcat_12['STRTTIME_first_trip'])
data_tripcat_12['ENDTIME_last_trip']  = data_tripcat_12['ENDTIME_last_trip'].fillna(data_tripcat_12['ENDTIME_first_trip'])
data_tripcat_12['TRPMILES_last_trip'] = data_tripcat_12['TRPMILES_last_trip'].fillna(data_tripcat_12['TRPMILES_first_trip'])

len(data_tripcat_12[data_tripcat_12.STRTTIME_last_trip.isna()])


# Last trip not to home
data_tripcat_12['WHYTO_last_trip'].value_counts()


#%%
####################   Create new dataframe 'data_tripcat_12_decihours'

#Convert start and end times into decimals: To be able to get statistics on them, we have to convert start and end times into decimals
#Problem: Some start/end times are 2-digits or less: 0, 6, 15, 30 (for HOUSEID 30004637 for example), as shown in the cell below. These times are in fact 00:00, 00:06, 00:15, 00:30. (checked by confirming with the travel minutes TRVLCMIN in the original data file) 

# investigating start and endtimes below 2 digits -> they are trips in the hour following midnight
# -> take this into account when converting the time format hhmm to decimal hours
data_tripcat_12[data_tripcat_12['STRTTIME_first_trip'] < 100].head(5)

# define the function to transform one HHMM input into decimal hours (will be used inside the dataframe with applymap function)
def from_hhmm_to_decimal_hours(input_number):
    mystring = str(input_number)
    m = float(mystring[-2:])
    if mystring[:-2] == '':
        h = float(0)
    else:
        h = float(mystring[:-2])
        
    decimal_hour = float(h) + float(m)/60
    
    #print(h, m)
    #print(decimal_hour)
    return(decimal_hour)

# testing the function
#from_hhmm_to_decimal_hours(130)

# convert the columns with start and end times to decimals
data_tripcat_12_decihour =  data_tripcat_12.copy()

data_tripcat_12_decihour[['STRTTIME_first_trip','ENDTIME_first_trip','STRTTIME_last_trip','ENDTIME_last_trip']] = \
    data_tripcat_12_decihour[['STRTTIME_first_trip','ENDTIME_first_trip','STRTTIME_last_trip','ENDTIME_last_trip']\
                                                     ].astype(int).applymap(from_hhmm_to_decimal_hours)

#Check if the decimals hours all make sense (observe min, max, mean, std)
data_tripcat_12_decihour.describe()

#%%
##################   Plot Histogram for start and end times

# histograms of start time of first trip and endtime of last trip

plt.figure(figsize=(5,5))

plt.title('Start time of first trip and end time of last trip', fontsize = 14, y = 1.05) #dpi = 300

# Data
plt.hist(data_tripcat_12_decihour['STRTTIME_first_trip'], bins = 100, color='grey', edgecolor='black', alpha=0.6, density=True, label='Start time of first trip')
plt.hist(data_tripcat_12_decihour['ENDTIME_last_trip'], bins = 100, color='black', edgecolor='black', alpha=0.6, density=True, label='End time of last trip')

# Axes
plt.xlim((0,24))
plt.ylim((0,0.3))
plt.grid(axis='y')
plt.xlabel('[h]')
plt.ylabel('Share of Vehicles')
           

plt.legend(loc='upper right')


plt.savefig('Start time of first trip and end time of last trip.jpeg', dpi = 300, transparent = False, bbox_inches="tight")
plt.show()


#%%
################### Calculate mena and std for first trips and last trips

#Calculate means of start time of first trip and end time of last trip
mean_STRTTIME_first_trip = ((data_tripcat_12_decihour['STRTTIME_first_trip'] * data_tripcat_12_decihour['WTTRDFIN'] / 365).sum()) \
                                / (data_tripcat_12_decihour['WTTRDFIN'] / 365).sum()

mean_ENDTIME_last_trip = ((data_tripcat_12_decihour['ENDTIME_last_trip'] * data_tripcat_12_decihour['WTTRDFIN'] / 365).sum()) \
                                / (data_tripcat_12_decihour['WTTRDFIN'] / 365).sum()

#Calculate standard deviation of start time of first trip and end time of last trip

std_STRTTIME_first_trip = ( \
                           (len(data_tripcat_12_decihour['STRTTIME_first_trip']) * (data_tripcat_12_decihour['WTTRDFIN'] * (data_tripcat_12_decihour['STRTTIME_first_trip'] - mean_STRTTIME_first_trip)**2).sum()) \
                           / ((len(data_tripcat_12_decihour['STRTTIME_first_trip']) - 1) * (data_tripcat_12_decihour['WTTRDFIN']).sum()) \
                           )**(1/2)

std_ENDTIME_last_trip = ( \
                           (len(data_tripcat_12_decihour['ENDTIME_last_trip']) * (data_tripcat_12_decihour['WTTRDFIN'] * (data_tripcat_12_decihour['ENDTIME_last_trip'] - mean_ENDTIME_last_trip)**2).sum()) \
                           / ((len(data_tripcat_12_decihour['ENDTIME_last_trip']) - 1) * (data_tripcat_12_decihour['WTTRDFIN']).sum()) \
                           )**(1/2)

print('===================================')
print('======= OUTPUTS FOR NETLOGO =======')
print('===================================')
print()

print('mean_STRTTIME_first_trip: ', mean_STRTTIME_first_trip)
print('mean_ENDTIME_last_trip: ', mean_ENDTIME_last_trip)
print('std_STRTTIME_first_trip: ', std_STRTTIME_first_trip)
print('std_ENDTIME_last_trip: ', std_ENDTIME_last_trip)


#%%
##################   Plot Histogram for start and end times
 
x = np.linspace (0, 200, 200)
appr_start_time = stats.norm.pdf(x, mean_STRTTIME_first_trip, std_STRTTIME_first_trip)
appr_end_time = stats.norm.pdf(x, mean_ENDTIME_last_trip, std_ENDTIME_last_trip)

plt.figure(figsize=(7,5))

plt.title('Start time of first trip and end time of last trip', fontsize = 14, y = 1.05) #dpi = 300

# Data
plt.plot(x, appr_start_time, '-y', label='Normal distribution of start time', color='red', alpha=0.6)
plt.plot(x, appr_end_time, '-y', label='Normal distribution of end time', color='blue', alpha=0.6)
plt.hist(data_tripcat_12_decihour['STRTTIME_first_trip'], bins = 100, color='grey', edgecolor='black', alpha=0.6, density=True, label='Start time of first trip')
plt.hist(data_tripcat_12_decihour['ENDTIME_last_trip'], bins = 100, color='black', edgecolor='black', alpha=0.6, density=True, label='End time of last trip')

# Axes
plt.xlim((0,24))
plt.ylim((0,0.3))
plt.grid(axis='y')
plt.xlabel('[h]')
plt.ylabel('Share of Vehicles')
plt.xticks(np.arange(0, 24, 4))

plt.legend(loc='upper right')


plt.savefig('Start time of first trip and end time of last trip.jpeg', dpi = 300, transparent = False, bbox_inches="tight")
plt.show()



#%%

################################   Output statictics to a text file    #######################################

# BUT EQUAL VALUES FOR ALL 4 CLUSTERS

# one column for the names
list_names = ['\'mean_ENDTIME_last_trip\'',
              '\'mean_STRTTIME_first_trip\'',
              '\'std_STRTTIME_first_trip\'',
              '\'std_ENDTIME_last_trip\'',
              '\'mean_daily_miles\'',
              '\'alpha_daily_miles\'',
              '\'lambda_daily_miles\'',
             ]

# one column for the values    
list_values = [mean_ENDTIME_last_trip,
               mean_STRTTIME_first_trip,
               std_STRTTIME_first_trip,
               std_ENDTIME_last_trip,
               mean_daily_miles,
               alpha_daily_miles,
               scale_daily_miles]

with open('driving_patterns.txt', mode='w') as txt_file:
    writer = csv.writer(txt_file, delimiter='\t')
    writer.writerows(zip(list_names, list_values))
    

#%%
    
########################################################################################
####################  Plotting nb vehicles parked vs driving over the day      #########
########################################################################################

#- DataFrame: 'data_tripcat_12_deciours, We use 'data_tripcat_12_decihours' because we want to avoid decimal numbers
#- List: vehicles_at_home, vehicles_not_at_home

def create_travel_patterns(df):
    my_list = np.array([0]*24)
    for index,row in df.iterrows():
        # housevehid = df.loc[index,'HOUSEVEHID']
        # initialize the driving pattern vector with 0s, 0 indicating that the vehicle is parked at home
        # my_dict[housevehid] = [0]*24
        
        start_time = df.loc[index,'STRTTIME_first_trip']
        end_time = df.loc[index,'ENDTIME_last_trip']
        
        # fill in the driving pattern vectors with 1s when the vehicle is "in driving pattern" (could be driving or parked)
        for i in range(int(start_time), int(end_time)):
            my_list[i] = my_list[i] + 1
                    
    return(my_list)   
    
#%%

#Create List with numbers of vehicles not at home during hours of the day
vehicles_not_at_home = create_travel_patterns(data_tripcat_12_decihour)

#Show list
vehicles_not_at_home 

#Create List with numbers of vehicles at home during hours of the day
nb_vehicles = len(data_tripcat_12_decihour)
vehicles_at_home = np.array
vehicles_at_home = nb_vehicles - vehicles_not_at_home

#PShow list
vehicles_at_home

#%%

## aggregate and plot the vector showing the number of vehicles driving every hour
x=range(0,24)
plt.title('Number of vehicles in driving pattern')
plt.xlabel('hour of the day')
plt.ylabel('vehicle units')

plt.stackplot(x, vehicles_at_home, vehicles_not_at_home, labels=['vehicles_at_home', 'vehicles_not_at_home'])

plt.legend(loc='lower center')
plt.show()

#%%
## get and plot the same information in "percentage of total vehicles" 

#Create Percentage Data
vehicles_not_at_home_perc = vehicles_not_at_home / nb_vehicles
vehicles_at_home_perc = vehicles_at_home / nb_vehicles

#Show list
vehicles_not_at_home_perc
vehicles_at_home_perc

#%%

#Create Plot
plt.figure(figsize=(7,5))

plt.title('Share of EVs at home', fontsize = 14, y = 1.05) #dpi = 300

x = range(0,24)

# Data
plt.stackplot(x, vehicles_at_home_perc * 100, vehicles_not_at_home_perc * 100, 
              labels = ['vehicles at home', 'vehicles driving, at work, or parked in public'],
              colors = ['black', 'grey'],
              alpha = 0.7)

# Axes
plt.xlim((0,23))
plt.ylim((0,100))
plt.grid(axis='y')
plt.xlabel('[h]')
plt.ylabel('[%]')
plt.xticks(np.arange(0, 24, 4))

plt.legend(loc='lower right')


plt.savefig('Share of EVs at home.jpeg', dpi = 300, transparent = False, bbox_inches="tight")
plt.show()




#%%

## Done here: confirm that this plot can lead to the charging behavior we see in the paper results 

# investigate maximum EV charging load at home, if all EVs are charging in every hour they are parked at home...
# with max L2 charging power 5kW, and 4.4 M EVs in CA 2030.
max_charging_power = 5
max_ev_charging_profile_at_home = np.array
max_ev_charging_profile_at_home = vehicles_at_home_perc * 4.4 * max_charging_power

print(max_ev_charging_profile_at_home)
print(vehicles_at_home_perc)

#%%
# plot maximum EV charging profile at home for 4.4 M vehicles (in GW)
fig, ax1 = plt.subplots(figsize=(7,5))


# Data
ax1.stackplot(x, vehicles_at_home_perc * 100, vehicles_not_at_home_perc * 100, 
              labels = ['vehicles at home', 'vehicles driving, at work or public'],
              colors = ['black', 'grey'],
              alpha = 0.7)

ax2 = ax1.twinx()
x_axis = range(0,24)
ax2.plot(x_axis, max_ev_charging_profile_at_home, 
        color='red',
        label = 'charging profile of EVs at home')



ax1.set_title('Share of EVs at home and max charging load', fontsize = 14, y = 1.05)
ax1.yaxis.grid(True)
ax1.set_xlim([0,23])
ax1.set_ylim([0,100])
ax1.set_xlabel('[h]', labelpad=10)
ax1.set_ylabel('vehicles [%]', labelpad=10)
#loc_y = plticker.MultipleLocator(base=4) 
#axes[0].yaxis.set_major_locator(loc_y)

ax2.set_ylabel('max charging load [GW]', labelpad=10)
#ax2.set_ylim([0,22])
#loc_y = plticker.MultipleLocator(base=4) 
#axes[0].yaxis.set_major_locator(loc_y)

fig.legend(loc='upper center', bbox_to_anchor=(0.52, 0.9), fancybox=False, edgecolor='grey')


plt.savefig('Share of EVs at home and max charging load.jpeg', dpi = 300, transparent = False, bbox_inches="tight")
plt.show()


#%%
#> (a) if every vehicle at home were charging at the maximum power, the minimum charging load occuring when there are the fewest vehicles at home (31.5%) would be 6.94 GW at 11 pm
#> (b) In our model results under the Hourly Pricing scenario, EV charging reaches a maximum load of 3.5 GW at 11am (see paper)
#> (c) this 3.5 GW load is by half inferior to the 7.0 GW upper bound at 11 am if all vehicles at home were charging in this hour.
#> Therefore the EV charging load results under Hourly Pricing are possible given the numbers of vehicles parked at home over the day: at this hour 11 am, half of the vehicles at home are charging.





