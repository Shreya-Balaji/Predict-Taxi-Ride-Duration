#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# The given task is to predict the total ride duration of taxi trip. The various data given include ID, VendorID, Pickup_DateTime, Dropoff_DateTime, NumberOfPassengers, Pickup_Lattitude, Pickup_Longitude, Dropoff_Lattitude, Dropoff_Lattitude, Store_and_FWD_flag, Trip_Duration. Interestingly, the pickup and date_time is not specified, which means that doesn't add any significance to the computations.

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[17]:


# Load the train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Display the first few rows of the train dataset
print(train_data.head())

# Display the first few rows of the test dataset
print(test_data.head())


# In[14]:


get_ipython().system('pip install haversine')


# In[21]:


# Convert date-time columns to proper data types for both train and test datasets
train_data['pickup_datetime'] = pd.to_datetime(train_data['pickup_datetime'])
test_data['pickup_datetime'] = pd.to_datetime(test_data['pickup_datetime'])

# Extract day of the week and hour of the day from pickup_datetime for both datasets
train_data['pickup_dayofweek'] = train_data['pickup_datetime'].dt.dayofweek
train_data['pickup_hour'] = train_data['pickup_datetime'].dt.hour

test_data['pickup_dayofweek'] = test_data['pickup_datetime'].dt.dayofweek
test_data['pickup_hour'] = test_data['pickup_datetime'].dt.hour
# Calculate distance between pickup and dropoff locations using Haversine formula for both datasets
from haversine import haversine

def calculate_distance(row):
    pickup_coords = (row['pickup_latitude'], row['pickup_longitude'])
    dropoff_coords = (row['dropoff_latitude'], row['dropoff_longitude'])
    return haversine(pickup_coords, dropoff_coords)

train_data['distance'] = train_data.apply(calculate_distance, axis=1)
test_data['distance'] = test_data.apply(calculate_distance, axis=1)


# In[22]:


plt.figure(figsize=(8, 6))
sns.histplot(train_data['trip_duration'], bins=50, kde=True)
plt.title('Distribution of Trip Duration')
plt.xlabel('Trip Duration (seconds)')
plt.ylabel('Frequency')
plt.show()

# Visualize the relationship between the pickup hour and average trip duration in the train dataset
plt.figure(figsize=(10, 6))
sns.barplot(x='pickup_hour', y='trip_duration', data=train_data, estimator=np.mean)
plt.title('Average Trip Duration by Pickup Hour')
plt.xlabel('Pickup Hour')
plt.ylabel('Average Trip Duration (seconds)')
plt.show()


# In[24]:


# Prepare the feature matrix X and target vector y for both datasets
X_train = train_data[['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
                     'dropoff_longitude', 'dropoff_latitude', 'pickup_dayofweek', 'pickup_hour', 'distance']]
y_train = train_data['trip_duration']

X_test = test_data[['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
                   'dropoff_longitude', 'dropoff_latitude', 'pickup_dayofweek', 'pickup_hour', 'distance']]

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# You can save the predictions to a new DataFrame and/or CSV file if needed
predictions_df = pd.DataFrame({'id': test_data['id'], 'trip_duration_predicted': y_pred})
predictions_df.to_csv('predictions.csv', index=False)


# In[25]:


# Read the predicted file
predicted_data = pd.read_csv('predictions.csv')

# Display the first few rows of the predicted DataFrame
print(predicted_data.head())


# In[26]:


# Suppose you have some input data with vendor_id, passenger_count, pickup_longitude, pickup_latitude,
# dropoff_longitude, dropoff_latitude, pickup_dayofweek, pickup_hour, and distance.

# Replace the following input values with your own data:
input_data = {
    'vendor_id': 1,
    'passenger_count': 2,
    'pickup_longitude': -74.008,
    'pickup_latitude': 40.730,
    'dropoff_longitude': -73.990,
    'dropoff_latitude': 40.755,
    'pickup_dayofweek': 3,  # Thursday (0: Monday, 1: Tuesday, ..., 6: Sunday)
    'pickup_hour': 15,
    'distance': 2.5  # Distance in kilometers
}

# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

# Use the trained model to predict trip duration for the input data
predicted_trip_duration = model.predict(input_df)

# Print the predicted trip duration
print("Predicted Trip Duration:", predicted_trip_duration[0])



