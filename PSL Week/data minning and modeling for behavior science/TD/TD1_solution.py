####################################################################
# This TD is using body data to explore :
#  - Panda dataframe, import .dat, header, separator, change key names, save csv, sorting dataframe, replacing values, drop nan
#  - Use of numpy : mean, median, max, min, var
#  - plotting histogram, density function, bimodal
#  - Confidence interval and boostrap computation, significance
#  - Create and call python functions
#  - Scatter plots, correlations, matrix of cross-correlation
#  - Simpson paradox, bar plot


################# ################# #################
################# Import packages
################# ################# #################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import choices












################# ################# #################
################# Open and format the data
################# ################# #################
dat =  pd.read_csv('./TD1/data_body/body.dat', sep='\s+', header=None) # Open the data file
Description = pd.read_csv('./TD1/data_body/description.csv', sep=';', header=0) # Open description.csv with description and names of each variables
dat = dat.rename(mapper=Description.loc[:,'Name'],axis=1) # Rename columns from dat with the name of the variable
dat.to_csv('./TD1/data_body/body.csv') # Save it to CSV so that we can look at it in excel.












################# ################# #################
################# Look at mean and median, min, max...
################# ################# #################
dat['Weight'] # Print the Weight of all people
np.mean(dat['Weight']) # Look at the average weight
np.median(dat['Weight']) # Look at the median weight

np.mean(dat['Height']) # Look at the average weight
np.median(dat['Height']) # Look at the median weight

dat['Sex'] # Print the sex of all people
np.mean(dat['Sex']) # Look at the average of sex => What does it correspond to ? = percentage of male in the dataset

np.max(dat['Biacromial']) # Look at the average Biacromial
np.min(dat['Biacromial']) # Look at the median Biacromial












################# ################# #################
################# Sorting the data
################# ################# #################
dat = dat.sort_values('Biacromial', ascending=True) # Sorting the data based on Biacromial value, ascending
plt.plot(np.array(dat['Biacromial']))
dat = dat.sort_values('Biiliac', ascending=False) # Sorting the data based on Biiliac value, descending
# Note that all other columns, including the first one containing the index has been re-arranged to match the new order.












################# ################# #################
################# Look at the variability of the data
################# ################# #################
plt.hist(dat['Weight'], 100) # Look at the histogram of the Weight : what do you notice ?
dat.loc[dat['Weight']>200,'Weight'] = np.nan # Replace the two outliers with Nan
plt.hist(dat['Weight'], 100) # PLot the histogram now that the two outliers are removed
np.var(dat['Weight'])

# Let's do the same with the Height
plt.hist(dat['Height'], 100) # Look at the histogram of the Weight : what do you notice ?
dat.loc[dat['Height']>1000,'Height'] = np.nan # Replace the two outliers with Nan
dat.loc[dat['Height']<50,'Height'] = np.nan # Replace the two outliers with Nan
plt.hist(dat['Height'], 100) # Plot the histogram now that the two outliers are removed
np.var(dat['Height'])

# Look at the density pLot
sns.kdeplot(dat['Weight'], bw=0.4) # Plot the kernel density plot with different kernel values
sns.kdeplot(dat['Height'], bw=0.4) # Plot the kernel density plot with different kernel values
sns.kdeplot(dat['Age'], bw=0.1) #Look at age distribution of this dataset

sns.kdeplot(dat['Elbow_diam'], bw=0.2) # Plot the kernel density plot of Elbow_diam with kernel value = 0.2 -> Bimodal
# Do not close the plot so that distribution will overlapp
sns.kdeplot(dat.loc[dat['Sex']==0,'Elbow_diam'], bw=0.2) # Now plot only form Men
sns.kdeplot(dat.loc[dat['Sex']==1,'Elbow_diam'], bw=0.2) # And for Women -> We see where the bimodal distribution is coming from


# Estimate confidence interval of the average using boostrap method
Weight =  np.array(dat['Weight'])
np.mean(Weight)
np.nanmean(Weight)
n_suj = np.shape(Weight)[0]
n_boostrap = 1000 # We will do 1000 boostrap itarations to estimate the mean
allmeans = []
for i in range(n_boostrap):
    allmeans.append(np.nanmean(choices(Weight, k=n_suj)))
sns.kdeplot(allmeans, bw=0.3) # PLot the distribution of estimated means
allmeans.sort() # Sort the list from low to high
allmeans = allmeans[int(0.025*n_boostrap) : n_boostrap - int(0.025*n_boostrap) ] # remove 2.5% lowest and 2.5% highest values
confidence_interval = [np.min(allmeans), np.max(allmeans)]
    # Note that his three last line can be replaced by the following function : np.percentile(allmeans, [2.5, 97.5])

# Now let's write a function that takes the name of the variable to consider, the index of subset of subject to consider, and the number of boostrap iteration and return the confidence interval of the mean
def Mean_boostrapCI(name,subjects, n_boot):
    var = np.array(dat.loc[subjects, name])
    allmeans = []
    for i in range(n_boot):
        allmeans.append(np.nanmean(choices(var, k=np.shape(var)[0])))
    confidence_interval = np.percentile(allmeans, [2.5, 97.5])
    return (confidence_interval)

# And use this function to compute the Weight CI male and female
male  = dat['Sex']==1
female  = dat['Sex']==0
Male_CI = Mean_boostrapCI('Weight',male,  1000)
Female_CI = Mean_boostrapCI('Weight',female,  1000)
    # -> What can be concluded on the significance of mean Weight diff between male and female ?
# re-use the same function to compute the CI of the mean Height between people < 40 yo and people > 40 yo
young  = dat['Age']<40
old  = dat['Age']>40
young_CI = Mean_boostrapCI('Height',young,  1000)
old_CI = Mean_boostrapCI('Height',old,  1000)
    # -> What can be concluded ?













################# ################# #################
################# Explore correlations between variables.
################# ################# #################
# For each subject plot its Weight and Height in a 2D plane
plt.scatter(dat['Weight'], dat['Elbow_diam']) # Scatter plot
corr = dat['Weight'].corr(dat['Elbow_diam']) # With numpy corr = np.corcoef(dat.loc[:, 'Weight'],dat.loc[:, 'Height'], rowvar = False)

# redo separately for Male/Female
plt.scatter(dat.loc[dat['Sex']==1, 'Weight'], dat.loc[dat['Sex']==1, 'Elbow_diam'])
plt.scatter(dat.loc[dat['Sex']==0, 'Weight'], dat.loc[dat['Sex']==0, 'Elbow_diam'])
corr_maleonly = dat.loc[dat['Sex']==1, 'Weight'].corr(dat.loc[dat['Sex']==1, 'Elbow_diam'])
corr_femaleonly = dat.loc[dat['Sex']==0, 'Weight'].corr(dat.loc[dat['Sex']==0, 'Elbow_diam'])
    # -> The correaltion coefficient reduced because part of the correaltion was already explained by biological sex.


# Look at the correlation between each pair of variables
variables = dat.keys()
n_var = np.shape(variables)[0]
correlations = np.zeros([n_var,n_var])
for i in range(n_var):
    for j in range(n_var):
        correlations[i, j] = dat[variables[i]].corr(dat[variables[j]])

# Now plot the matrix of cross correlation with colorbar and variables on the axis.
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations)
plt.xticks(range(0,n_var))
plt.yticks(range(0,n_var))
ax.tick_params(axis='x', rotation=90)
ax.set_xticklabels(variables)
ax.set_yticklabels(variables)
cbar = fig.colorbar(cax)
















################# ################# #################
################# The Simpson paradox
################# ################# #################
# Open the data
vacc =  pd.read_csv('./TD1/data_vaccine/Data_vaccine.csv', sep=';', header=0) # Open the data file

# Look at the mortality for people with and without vaccine :
with_vacc = np.mean(vacc.loc[vacc['Vaccine']==1, 'Deceased'])
without_vacc = np.mean(vacc.loc[vacc['Vaccine']==0, 'Deceased'])
plt.bar(['Vacc', 'noVacc'], [with_vacc,without_vacc], color=['g', 'b'])
    # -> Does it mean that the vaccine is dangerous ?

# But now, look at the age of people and their vaccination rate
vacc_rate_young = np.mean(vacc.loc[vacc['Age ']=='<60', 'Vaccine'])
vacc_rate_old = np.mean(vacc.loc[vacc['Age ']=='>60', 'Vaccine'])
# But now, look at mortality as a function of age
mortality_rate_young = np.mean(vacc.loc[vacc['Age ']=='<60', 'Deceased'])
mortality_rate_old = np.mean(vacc.loc[vacc['Age ']=='>60', 'Deceased'])

# So if we look at mortality by age and vacc status
mortality_rate_young_vacc = np.mean(vacc.loc[(vacc['Vaccine']==1) & (vacc['Age ']=='<60'), 'Deceased'])
mortality_rate_young_novacc = np.mean(vacc.loc[(vacc['Vaccine']==0) & (vacc['Age ']=='<60'), 'Deceased'])
mortality_rate_old_vacc = np.mean(vacc.loc[(vacc['Vaccine']==1) & (vacc['Age ']=='>60'), 'Deceased'])
mortality_rate_old_novacc = np.mean(vacc.loc[(vacc['Vaccine']==0) & (vacc['Age ']=='>60'), 'Deceased'])

plt.bar(['Young_Vacc', 'Young_noVacc','Old_Vacc', 'Old_noVacc'], [mortality_rate_young_vacc,mortality_rate_young_novacc,mortality_rate_old_vacc,mortality_rate_old_novacc ], color=['g', 'b','g', 'b'])
    # -> This is an example of a confounding factor creating the Simpson Paradox.





