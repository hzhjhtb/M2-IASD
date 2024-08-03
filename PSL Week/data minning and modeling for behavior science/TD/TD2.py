####################################################################
# This TD is using body data to explore :
#  - Linear regression
#  - Multiple linear regression
#  - Ridge / Lasso

################# ################# #################
################# Import packages
################# ################# #################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import choices
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Open the data
dat =  pd.read_csv('./TD1/data_body/body.dat', sep='\s+', header=None) # Open the data file
Description = pd.read_csv('./TD1/data_body/description.csv', sep=';', header=0) # Open description.csv with description and names of each variables
dat = dat.rename(mapper=Description.loc[:,'Name'],axis=1) # Rename columns from dat with the name of the variable

# Remove outliers
dat.loc[dat['Weight']>200,'Weight'] = np.nan # Replace the two outliers with Nan
dat.loc[dat['Height']>1000,'Height'] = np.nan # Replace the two outliers with Nan
dat.loc[dat['Height']<50,'Height'] = np.nan # Replace the two outliers with Nan


################# ################# #################
################# Simple Linear regression
################# ################# #################
# Run a linear regression between two variables
linear_reg = LinearRegression(fit_intercept=False)
X = dat[['Weight']]
Y = dat['Height']
linear_reg.fit(X,Y)
    # -> Won't work because of nan values => Need to clean the dataframe from incomplete data

# Q1 Remove subjects with nan values
dat = XXXXXXXXX
X = dat[['Weight']] # Extract Weights
Y = dat['Height'] # Extract Heights
linear_reg.fit(X,Y) # Fit the linear regression model

# Q2 print the coefficient of this regression
XXXXXXXXX

# Q3 make a scatter plot of Weight and Height and superimpose the linear regression
plt.XXXXXXXXX # Scatter plot
plt.plot(X, XXXXXXXXX, 'k') # Draw linear regression on top

# Q4 -> What is the problem with this regression ?


# Q5 -> Redo the same but corrcting the problem of the previous regression
linear_reg = LinearRegression(XXXXXXXXX)
X = dat[['Weight']]
Y = dat['Height']
linear_reg.fit(X,Y)

# Q6 look at the intercept and the coef of this model
intercept = XXXXXXXXX
coef = XXXXXXXXX

# Q7 make a scatter plot of Weight and Height and superimpose the linear regression
XXXXXXXXX # Scatter plot
plt.plot(X, XXXXXXXXX, 'k') # Draw linear regression on top



################# ################# #################
################# Multiple Linear regression
################# ################# #################
# Q8 Before computing regression with multiple factors, we need to zscore all factors so that they are comparable.
dat = XXXXXXXXX


# Q9  Make a regression of the Height with two factors : Height and Knee-girth
linear_reg = XXXXXXXXX
X = XXXXXXXXX
Y = dat['Height']
linear_reg.fit(X,Y)

# Q10 make a barplot showing the two coefficients of this regression
plt.bar([1,2], linear_reg.coef_)

# Q11 How stable this is ? Compute confidence interval on the regression parameters using a boostrap approach. Create the loop to compute 1000 estimation of each coefs
XXXXXXXXXXXXXX

# Q12 From these 1000 estimation, compute the confidence interval for both coefs
CI1 = XXXXXXXXXXXXXX
CI2 = XXXXXXXXXXXXXX

# Q13 redo the bar plot and add the confidence interval
plt.bar([1,2], XXXXXXXXXXXXXX)
plt.errorbar([1,2], XXXXXXXXXXXXXX, yerr=XXXXXXXXXXXXXX, fmt="o", color="r")


# Q14 Know look only at the simple regression between Height and Knee_girth
XXXXXXXXXXXXXX

# Q15 -> What can you conclude ?


# Q16 Now make a multiple linear regression predicting the Height based on all other factors.
linear_reg = LinearRegression(fit_intercept=True)
X = XXXXXXXXXXXXXX
Y = dat['Height']
linear_reg.fit(X,Y)

# Q17 And plot all coefs of this regression
XXXXXXXXXXXXXX


# Q18 Redo the same but with using a ridge regression with alpha = 100
ridge_reg = XXXXXXXXXXXXXX
X = XXXXXXXXXXXXXX
Y = dat['Height']
ridge_reg.fit(X,Y)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(X.keys(), XXXXXXXXXXXXXX)
ax.tick_params(axis='x', rotation=90)

# Q19 what do you observe ?

# Q20 Compute the values of the parameters for ridge regressions with alpha varying between 10^-5 and 10^5
X = XXXXXXXXXXXXXX
Y = dat['Height']
alphas = np.logspace(XXXXXXXXXXXXXX, XXXXXXXXXXXXXX, num=1000)
for alpha in XXXXXXXXXXXXXX:
    XXXXXXXXXXXXXX

plt.xscale("log")
plt.plot(alphas,allcoefs)


# Q21 Now compute a lasso regresson with alpha = 0.1
lasso_reg = XXXXXXXXXXXXXX
X = XXXXXXXXXXXXXX
Y = dat['Height']
lasso_reg.fit(X,Y)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(X.keys(), XXXXXXXXXXXXXX)
ax.tick_params(axis='x', rotation=90)


# Q22 Compute and plot the values of the parameters for lasso regressions with alpha varying between 10^-5 and 10^5
XXXXXXXXXXXXXX