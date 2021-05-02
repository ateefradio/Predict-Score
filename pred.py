import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.linear_model
import sklearn.preprocessing
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("preprocessed.csv")
data.head()
# data=data.drop(columns=['Unnamed:0'])
data.drop("Unnamed: 0",axis=1)
from datetime import datetime
data['start_date'] = data['start_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
# Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=data, columns=['batting_team', 'bowling_team'])
# Splitting the data into train and test set
X_train = encoded_df.drop(labels='total_runs', axis=1)[encoded_df['start_date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total_runs', axis=1)[encoded_df['start_date'].dt.year >= 2018]
y_train = encoded_df[encoded_df['start_date'].dt.year <= 2016]['total_runs'].values
y_test = encoded_df[encoded_df['start_date'].dt.year >= 2018]['total_runs'].values
# Removing the 'date' column
X_train.drop(labels='start_date', axis=True, inplace=True)
X_test.drop(labels='start_date', axis=True, inplace=True)

# --- Model Building ---
# Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
LinearRegressor=LinearRegression()

LinearRegressor.fit(X_train,y_train)
# Creating a pickle file for the classifier
import pickle
filename = 'model.pkl'
pickle.dump(regressor, open(filename, 'wb'))
