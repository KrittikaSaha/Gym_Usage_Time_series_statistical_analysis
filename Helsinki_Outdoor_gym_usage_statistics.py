import pandas as pd
import numpy as np

###################################################################################################
#                                     Task 1                                                      #
###################################################################################################
gym_10_min_df = pd.read_csv('hietaniemi-gym-data.csv')

#convert time to datetime
gym_10_min_df['time']=pd.to_datetime(gym_10_min_df['time'])

#resample hourly with sum
gym_hourly_df = gym_10_min_df.resample('60min', on='time').sum().reset_index()

#gym_hourly_df['hour']= gym_hourly_df.time.dt.hour
#present first 10 rows
print(f'first 10 rows of hourly sampled data: {gym_hourly_df.head(10)}')



###################################################################################################
#                                     Task 3                                                      #
###################################################################################################

# most popular device
def calculate_max_usage(df):
    total_usage=[]
    for col in df.columns[1:-1]:
        total_usage.append({'sensor':col, 'usage_minutes': df[col].sum()})
    
    max_usage_minutes = max(total_usage, key=lambda x: x['usage_minutes'])
    max_usage_sensor = max_usage_minutes['sensor']
    print(f'Most popular device: {max_usage_sensor}')
    return max_usage_sensor

most_popular = calculate_max_usage(gym_10_min_df)
print(f""" Most popular device was {most_popular} with total {most_popular['usage_minutes']} minutes spent""")

##################################################################################################################################################

# impact of hour on popularity of outer gym

hourly_usage= gym_hourly_df.groupby('hour').sum()

# transpose the DataFrame
df_transposed = hourly_usage.T

# calculate the sum of each row
df_transposed['sum'] = df_transposed.sum(axis=1)
most_popular_hour = df_transposed['sum'].idxmax()
max_minutes= df_transposed['sum'].max()
print(f'Most populous hour was {most_popular_hour}:00 hrs with total {max_minutes} minutes spent')

##################################################################################################################################################

# impact of weekend on popularity of outer gym

# add a new column with the weekday names
gym_hourly_df['weekday'] = gym_hourly_df['time'].dt.strftime('%A')
weekday_usage = gym_hourly_df.groupby('weekday').sum().drop(columns='hour')

# avg weekday usage
def calculate_mean_usage(df):
    mean_usage={}
    for col in df.columns:
        mean_usage[col]=df[col].mean()
        #mean_usage.append({'Weekday':col, 'usage_minutes': df[col].mean()})
    return mean_usage    

mean_usage = calculate_mean_usage(weekday_usage.T)

avg_wknd_usage = np.mean([mean_usage['Saturday'],mean_usage['Sunday']])
avg_wkday_usage = np.mean([v for k, v in mean_usage.items() if k not in ('Saturday', 'Sunday')])

print(f'Avg weekday usage: {avg_wkday_usage}, whereas avg weekend usage: {avg_wknd_usage}, hence gym was not more popular on weekends')

        

###################################################################################################
#                                     Task 4                                                      #
###################################################################################################       


# add a new column with the weekday as a number
gym_10_min_df['weekday'] = gym_10_min_df['time'].dt.weekday

# add a new column with the hour as a number
gym_10_min_df['Hour'] = gym_10_min_df['time'].dt.strftime('%H:%M')

# Sum of minutes across all gym devices
gym_10_min_df['sum'] = gym_10_min_df.drop(['time','weekday','Hour'], axis=1).sum(axis=1)

gym_10_min_df.head(10)




gym_hourly_df['weekday'] = gym_hourly_df['time'].dt.weekday

# add a new column with the hour as a number
gym_hourly_df['hour'] = gym_hourly_df['time'].dt.strftime('%H:%M')

# Sum of minutes across all gym devices
gym_hourly_df['sum'] = gym_hourly_df.drop(['time','weekday','hour'], axis=1).sum(axis=1)

gym_hourly_df.head(10)
###################################################################################################
#                                     Task 5.1 simple corr                                        #
###################################################################################################    

#impact of weather on gym popularity

kaisaniemi_weather_data = pd.read_csv('kaisaniemi-weather-data.csv')
kaisaniemi_weather_data.head(10)

# combine columns and add timezone information

kaisaniemi_weather_data['timestring'] = kaisaniemi_weather_data.apply(lambda x: f"{x['Year']}-{x['Month']:02}-{x['Day']:02} {x['Hour']}", axis=1)
# Convert the datetime string column to datetime64 format with timezone information
kaisaniemi_weather_data['time'] = pd.to_datetime(kaisaniemi_weather_data['timestring']).astype('datetime64[ns, UTC]')


################################################################

# merge the weather data with hourly gym usage on time column
merged_df = pd.merge(gym_hourly_df, kaisaniemi_weather_data.drop(columns='Hour'), on='time', how='left')


################################################################
# merge with 10 min aggregates instead of hourly
# resample weather data with 10 min interval. The ffill method is used to forward-fill missing values, which fills missing values with the most recently observed value.
kaisaniemi_weather_data.set_index('time', inplace=True)
kaisaniemi_weather_10_min = kaisaniemi_weather_data.drop(columns=['timestring','Timezone']).resample('10T').ffill().reset_index()
#kaisaniemi_weather_10_min['Hour'] = kaisaniemi_weather_10_min['time'].dt.strftime('%H:%M')

# merge the weather data with 10 min gym usage on time column
merged_10min_df = pd.merge(gym_10_min_df, kaisaniemi_weather_10_min.drop(columns='Hour'), on='time', how='left')

################################################################

# The following analysis on the rest of the script is done using hourly aggreagate merged data, so as to not use data with noise introduced by ffill

# impact of temperature on popularity of outer gym
corr_temp = merged_df['Temperature (degC)'].corr(merged_df['sum'])

# print the correlation coefficient
print(f'Correlation coefficient: {corr_temp:.2f}')
# A correlation coefficient of 0.33 suggests a moderate positive correlation 
# between the two variables being compared. However, it is important to keep in mind 
# that correlation does not imply causation and there could be other factors at play t
# hat are affecting the relationship between the variables.

###########################################################################################

#impact of snow 

merged_df['Snow'] = merged_df['Snow depth (cm)'].apply(lambda x: 'No Snow' if x == -1 else 'Snow Possible' if x == 0 else 'Snow')

# convert Snow column into dummy variables
snow_dummies = pd.get_dummies(merged_df['Snow'], prefix='Snow')

# concatenate dummy variables with original dataframe
merged_df = pd.concat([merged_df, snow_dummies], axis=1)

# calculate correlation between snow dummies and gym_usage column
corr_snow = merged_df[['Snow_No Snow', 'Snow_Snow Possible', 'Snow_Snow', 'sum']].corr()

print(corr_snow)
# The 'Snow_No Snow' column has a positive correlation of 1 with itself (as expected).
# The 'Snow_Snow Possible' column has a negative correlation of 0.63 with the 'Snow_No Snow' column, 
# indicating that there is a moderate negative relationship between the presence of possible snow and gym usage.
# The 'Snow_Snow' column has a negative correlation of 0.43 with the 'Snow_No Snow' column, indicating that there is a moderate negative relationship between the presence of actual snow and gym usage.
# The 'sum' column has a positive correlation of 0.11 with the 'Snow_No Snow' column, indicating a weak positive relationship between gym usage and the absence of snow. 
# However, the correlation is very weak and suggests that there is not a strong relationship between these variables.




###################################################################################################
#                                     Task 5.2 Linear Regressor                                     #
###################################################################################################   

# ML

# move column 'sum' to the last position
#new_order = [col for col in merged_df.columns if col != 'sum'] + ['sum']
#merged_df = merged_df.reindex(columns=new_order)
#merged_df.drop(columns=['Timezone','timestring']).to_csv('merged.csv')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

nan_cols = merged_df.columns[merged_df.isna().any()].tolist()

# create an imputer object with the desired strategy (e.g., mean, median, most_frequent, constant)
imputer = SimpleImputer(strategy='mean')


# transform the DataFrame to replace NaN values with the imputed values
merged_df['precp'] = imputer.fit_transform(merged_df[['Precipitation (mm)']])
merged_df['snowd'] = imputer.fit_transform(merged_df[['Snow depth (cm)']])
merged_df['temp'] = imputer.fit_transform(merged_df[['Temperature (degC)']])


# Prepare your dataset
X = merged_df[['hour', 'weekday', 'precp', 'snowd','Snow_No Snow', 'Snow_Snow Possible', 'Snow_Snow', 'temp']]
y = merged_df['sum']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose and train a machine learning model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate your machine learning model
y_pred = model.predict(X_test)


from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error

# y_test and y_pred are the true and predicted target values respectively
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Explained Variance Score: {evs}")
print(f"Mean Absolute Error: {mae}")

#Based on the above metrics the model's performance is not ideal. 
# The MSE value of 3132.97 indicates that there is a high variance between the predicted values and actual values. 
# The R-squared value of 0.166 indicates that the model is not able to explain much of the variance in the data. 
# The explained variance score of 0.167 suggests that the model's predictions are not very accurate. 
# The mean absolute error of 43.69 indicates that, on average, the predicted values are off by approximately 43.69 units. 
# Overall, these metrics suggest that the model may need further tuning or a different approach may need to be considered.

# Get feature importance
importance = model.coef_

# Create a dataframe to store the feature importance values
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})

# Sort the dataframe by feature importance in descending order
features_df = features_df.sort_values('Importance', ascending=False)

# Print the feature importance values
print(features_df)


###################################################################################################
#                                     Task 5.3 Random Forest Regressor                            #
###################################################################################################   

from sklearn.ensemble import RandomForestRegressor

# Prepare your dataset
X = merged_df[['hour', 'weekday', 'precp', 'Snow_No Snow', 'Snow_Snow Possible', 'Snow_Snow', 'temp']]
y = merged_df['sum']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose and train a machine learning model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate your machine learning model
y_pred = model.predict(X_test)

# y_test and y_pred are the true and predicted target values respectively
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Explained Variance Score: {evs}")
print(f"Mean Absolute Error: {mae}")

# The performance of the random regressor is not good as the mean squared error is high 
# and the R-squared and explained variance score are low. 
# The mean absolute error is also quite high. This indicates that the model is not able to 
# capture the relationship between the features and the target variable and is essentially making random predictions.

# Get feature importance
importance = model.coef_

# Create a dataframe to store the feature importance values
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})

# Sort the dataframe by feature importance in descending order
features_df = features_df.sort_values('Importance', ascending=False)

# Print the feature importance values
print(features_df)


###################################################################################################
#                               Task 5.4 Gradient Boosting Regressor                              #
###################################################################################################

from sklearn.ensemble import GradientBoostingRegressor

# Prepare your dataset
X = merged_df[['hour', 'weekday', 'precp', 'Snow_No Snow', 'Snow_Snow Possible', 'Snow_Snow', 'temp']]
y = merged_df['sum']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose and train a machine learning model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate your machine learning model
y_pred = model.predict(X_test)


# y_test and y_pred are the true and predicted target values respectively
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Explained Variance Score: {evs}")
print(f"Mean Absolute Error: {mae}")

#The performance metrics for the Gradient Boosting model seem to be better than those 
# for the Random Forest model and the Linear Regression model. 
# The lower Mean Squared Error (MSE) and Mean Absolute Error (MAE) 
# indicate that the model's predictions are closer to the actual values. 
# The higher R-squared and Explained Variance Score indicate that the model is able to explain more of the variance in the data.


# Get feature importance
importance = model.coef_

# Create a dataframe to store the feature importance values
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})

# Sort the dataframe by feature importance in descending order
features_df = features_df.sort_values('Importance', ascending=False)

# Print the feature importance values
print(features_df)


# Based on the feature importance values, it appears that the most important features for predicting gym usage are
# 'Snow_Snow Possible', 'Snow_Snow', and 'temp', with 'Snow_Snow Possible' being the most important. 
# The feature 'Snow_No Snow' appears to have a negative impact on the model's performance, indicating that the presence of snow may discourage gym usage.
# 
# The feature importance values can be interpreted as follows:
# 
# A positive value indicates that the feature has a positive impact on the model's predictions, meaning that an increase in that feature's value is associated with an increase in gym usage.
# A negative value indicates that the feature has a negative impact on the model's predictions, meaning that an increase in that feature's value is associated with a decrease in gym usage.
# The larger the absolute value of the feature importance, the more impact the feature has on the model's predictions.


###################################################################################################
#                                     Task 6                                                      #
###################################################################################################    

import joblib
model = joblib.load('model.pkl')

# Weekday as integer
merged_df['weekday'] = merged_df['weekday'].astype(int)
# hour as integer
merged_df['hour'] = merged_df.time.dt.hour.astype(int)

# Precipitation in millimeters as float
merged_df['Precipitation (mm)'] = merged_df['Precipitation (mm)'].astype(float)

# Snow depth in centimiters as float
merged_df['Snow depth (cm)'] = merged_df['Snow depth (cm)'].astype(float)

# Temperature in Celsius as float
merged_df['Temperature (degC)'] = merged_df['Temperature (degC)'].astype(float)

# reindex 

new_order = ['weekday','hour','Precipitation (mm)','Snow depth (cm)','Temperature (degC)']
test = merged_df.reindex(columns=new_order)

model.predict(test.dropna())


################# imputer ############

#nan_mask = test.isna().any(axis=1)
#nan_rows = test[nan_mask]

from sklearn.impute import SimpleImputer

model = joblib.load('model.pkl')

# create an imputer object with the desired strategy (e.g., mean, median, most_frequent, constant)
imputer = SimpleImputer(strategy='mean')

df_imputed = merged_df.copy()
# transform the DataFrame to replace NaN values with the imputed values
df_imputed['Precipitation (mm)'] = imputer.fit_transform(merged_df[['Precipitation (mm)']])
df_imputed['Snow depth (cm)'] = imputer.fit_transform(merged_df[['Snow depth (cm)']])
df_imputed['Temperature (degC)'] = imputer.fit_transform(merged_df[['Temperature (degC)']])

df_imputed = df_imputed[['weekday','hour','Precipitation (mm)','Snow depth (cm)','Temperature (degC)']]

# print the result
print(df_imputed)

# predictions
df_imputed['predictions']=model.predict(df_imputed)

dataset_with_predictions = gym_hourly_df.join(df_imputed['predictions'])

processed_dataset_with_predictions = merged_df.join(df_imputed['predictions'])[['time', '19', '20', '21', '22', '23', '24', '25', '26', 'hour',
       'weekday', 'Year', 'Month', 'Day', 'Timezone',
       'Precipitation (mm)', 'Snow depth (cm)', 'Temperature (degC)','sum', 'predictions']]


