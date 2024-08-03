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
# If you are not familiar with Python : this part allow to import packages functions.
# You can either import one function (from Package import Function) or the full package (import package as pkg) and call eah function of this package as pkg.function
import pandas as pd              # Pandas : see https://pandas.pydata.org/docs/user_guide/10min.html
import numpy as np               # Numpy : see https://numpy.org
import matplotlib.pyplot as plt  # matplotlib for visualization : https://matplotlib.org
import seaborn as sns            # seaborn is another plotting library : https://seaborn.pydata.org
from random import choices


################# ################# #################
################# Open and format the data
################# ################# #################
    # -> Using pandas, you can open .dat or .csv data files
dat =  pd.read_csv('./TD1/data_body/body.dat', sep='\s+', header=None) # Open body.dat using read_csv function from pandas library.

## Q1 ##  Using the same read_csv function from panda, open the description.csv file. Choose the correct 'sep' and 'header' parameters
Description = XXXXXXXXXX
dat = dat.rename(mapper=Description.loc[:,'Name'],axis=1) # What is the role of this line ?

## Q2 ## save the dataframe as csv.
XXXXXXXXXX


################# ################# #################
################# Look at mean and median, min, max...
################# ################# #################
dat['Weight'] # Print the Weight of all people
## Q3 ## Using a function from numpy, compute the mean Weight
mean_weight = XXXXXXXXXX
## Q4 ## Using a function from numpy, compute the median Weight
median_weight = XXXXXXXXXX

## Q5 ## Redo the same for the height
mean_weight = XXXXXXXXXX
median_weight = XXXXXXXXXX

## Q6 ## Now select the column corresponding to biological sex of subjects
dat[XXXXXXXXXX]
## Q7 ## And look at its average value => What does it correspond to ? : XXXXXXXXXXXXXX
Mean_sex = XXXXXXXXXX

    # -> With numpy, you can alo find the Maximum / Minimum
np.max(dat['Biacromial']) # Max Biacromial
np.min(dat['Biacromial']) # Min Biacromial


################# ################# #################
################# Sorting the data
################# ################# #################
    # -> Pandas dataframe can be sorted based of a given column. Find the correct method to apply on a dataframe to do so.
## Q7 ## First sort the based on Biacromial value, ascending and plot those values. Plot another column of the dataframe.
XXXXXXXXXX('Biacromial', ascending=True)
## Q8 ## And now base on the Biiliac, descending.
XXXXXXXXXX


    # ->  Note that all other columns, including the first one containing the index has been re-arranged to match the new order.



################# ################# #################
################# Look at the variability of the data
################# ################# #################
## Q9 ## Look at the doc from matplotlib.pyplot to plot the histogram of Weight with 100 bins.
XXXXXXXXXX # -> What do you notice ?
## Q10 ## This command will find the outliers : 'dat['Weight']>200'. Use this to replace the outliers by nans in the dataframe
XXXXXXXXXX = np.nan  # Replace the two outliers with Nan
## Q11 ## Now replot the hitogram and check that outliers are removed
XXXXXXXXXX
np.var(dat['Weight'])


## Q12 ## Redo the same process with Height to plot the histogram without outliers
XXXXXXXXXXX

## Q13 ## barplot are interesting but if we want to see a continuous distribution, density plot are interesting. Find a use a function from seaborn package to compute density plot of the Weight/Height and age
XXXXXXXXXX
XXXXXXXXXX
XXXXXXXXXX

## Q14 ## Play with the kernel size and observe how ot changes the distribution plot.
XXXXXXXXXX



## Q15 ## Plot the diameer of elbows with a kernel size of 1
XXXXXXXXXX

## Q16 ## Redo the same with a kerel size of 0.2 : What do you notice ?
XXXXXXXXXX

## Q17 ## If you do not close the figure between two plots, they will overlapp. Use this to overlapp the density plt of elbow diameters of the whole population with the one with only males and the one with only females (alway with kernel width = 0.2). : What can you conclude ?
XXXXXXXXXX
XXXXXXXXXX
XXXXXXXXXX

## Q18 ## Now we will estimate the confidence intervall of the weight using a boostrap method.
Weight =  np.array(dat['Weight'])
mean = XXXXXXXXXX
n_suj = np.shape(Weight)[0]
n_boostrap = 1000  # We will do 1000 boostrap iterations to estimate the mean
allmeans = [] # In this list, we will put every estimation of the mean for each boostrap
for i in range(n_boostrap):
    sample = XXXXXXXXXX
    allmeans.XXXXXXXX(sample)

## Q19 ## plot the density plot of all means
sns.kdeplot(allmeans, bw=0.3) # PLot the distribution of estimated means

## Q20 ## Find the 2.5% lowest and 2.5% highest values of allmean :
XXXXXXXXXX #  fisrt : sort the list from low to high
allmeans = allmeans[XXXXXXXXXX] # Slice the array to remove the 2.5% lowest and the 2.5% highest values
confidence_interval = [XXXXXXXXXX, XXXXXXXXXX] # Find the 95% CI
    # ->  Note that this three last line can be replaced by the following function : np.percentile(allmeans, [2.5, 97.5])


## Q21 ## Now you will write a function that takes the name of the variable to consider, the index of subset of subject to consider, and the number of boostrap iteration and return the confidence interval of the mean
def Mean_boostrapCI(name,subjects, n_boot):
    variable  = XXXXXXXXXX
    allmeans = []
    for i in range(n_boot):
        allmeans = XXXXXXXXXX
    confidence_interval = [XXXXXXXXXX,XXXXXXXXXX] # find CI
    return (confidence_interval)


## Q22 ## Use this function to compute the Weight CI male and female
male  = dat['Sex']==1
female  = dat['Sex']==0
Male_CI = XXXXXXXXXX
Female_CI = XXXXXXXXXX
## Q23 ## What can be concluded on the significance of mean Weight diff between male and female ? : XXXXXXXXXX


## Q24 ## re-use the same function to compute the CI of the mean Height between people < 40 yo and people > 40 yo
young  = XXXXXXXXXX
old  = XXXXXXXXXX
young_CI = XXXXXXXXXX
old_CI = XXXXXXXXXX
## Q25 ## What can you conclude ?



################# ################# #################
################# Explore correlations between variables.
################# ################# #################
## Q26 ##  Using the scatter function from matplotlib.pyplot, show Height and Weight for each subject
XXXXXXXXXX

## Q27 ##  Compute the correlation between the two variables Height and Weight
corr = XXXXXXXXXX

## Q28 ## Redo the same for male and female separately
plt.scatter(XXXXXXXXXX)
plt.scatter(XXXXXXXXXX)
corr_maleonly = XXXXXXXXXX
corr_femaleonly = XXXXXXXXXX


## Q29 ## Write loops to look at the correlation of each pair of variable
variables = dat.keys() # Find all keys in dataframe
n_var = np.shape(variables)[0] # number of keys in dataframe
correlations = np.zeros([n_var,n_var]) # Create a matrix full of zeros to put the correlations
XXXXXXXXXX

## Q30 ## Plot the Matrix of correlations of all variables
fig = plt.figure()
ax = fig.add_subplot()
cax = ax.XXXXXXXXXX
plt.xticks(range(0,n_var)) # Add a tick for each variable
plt.yticks(range(0,n_var))
ax.tick_params(axis='x', rotation=90) # Add tilt to ticks
ax.set_xticklabels(variables)
ax.set_yticklabels(variables)
cbar = fig.colorbar(cax)






################# ################# #################
################# The Simpson paradox
################# ################# #################
## Q31 ## Open the Data_vaccine.csv file and save it in .csv format
vacc =  XXXXXXXXXX

## Q32 ## Look at the mortality rate of vaccinated and non-vaccinated people
with_vacc = XXXXXXXXXX
without_vacc = XXXXXXXXXX
plt.bar(['Vacc', 'noVacc'], [with_vacc,without_vacc], color=['g', 'b'])
    # -> What do you conclude ? Is the vaccine really dangerous ?


## Q33 ## Now, look at vaccination rate as a function of age
vacc_rate_young = XXXXXXXXXX
vacc_rate_old = XXXXXXXXXX

## Q34 ## And look at mortality rate as a function of age
mortality_rate_young = XXXXXXXXXX
mortality_rate_old = XXXXXXXXXX

## Q35 ## make a bar plot with mortality rate separated by vaccination status and age
mortality_rate_young_vacc = XXXXXXXXXX
mortality_rate_young_novacc = XXXXXXXXXX
mortality_rate_old_vacc = XXXXXXXXXX
mortality_rate_old_novacc = XXXXXXXXXX

plt.bar(['Young_Vacc', 'Young_noVacc','Old_Vacc', 'Old_noVacc'], [mortality_rate_young_vacc,mortality_rate_young_novacc,mortality_rate_old_vacc,mortality_rate_old_novacc ], color=['g', 'b','g', 'b'])
    # -> This is an example of a confounding factor creating the Simpson Paradox (https://fr.wikipedia.org/wiki/Paradoxe_de_Simpson)





""