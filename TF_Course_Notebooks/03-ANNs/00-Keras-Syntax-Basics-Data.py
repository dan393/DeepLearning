import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/dan39/OneDrive/Desktop/DeepLearning/tensorflow/TF_Course_Notebooks/DATA/kc_house_data.csv')
df.isnull().sum()

plt.figure(figsize=(10, 6))
sns.distplot(df['price'])
plt.show()

sns.countplot(df['bedrooms'])
plt.show()

df.corr()['price'].sort_values()
plt.figure(figsize=(10, 5))
sns.scatterplot(x='price', y='sqft_living', data=df)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxenplot(x='bedrooms', y='price', data=df)
plt.show()

plt.figure(figsize=(12, 8))
sns.scatterplot(x='price', y='long', data=df)
plt.show()

plt.figure(figsize=(12, 8))
sns.scatterplot(x='long', y='lat', data=df, hue='price')
plt.show()

df.sort_values('price', ascending=False).head(20)
plt.show()

non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]

plt.figure(figsize=(12, 8))
sns.scatterplot(x='long', y='lat', data=non_top_1_perc, edgecolor=None, alpha=0.2, palette='RdYlGn', hue='price')
plt.show()

sns.boxenplot(x='waterfront', y='price', data=df)
plt.show()

df = df.drop('id', axis=1)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)

X = df.drop('price', axis=1).values
y = df['price'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X_train.shape

model = Sequential()
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss = 'mse')

model.fit(x=X_train, y = y_train, validation_data = (X_test, y_test))

losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

from sklearn.metrics import  mean_squared_error, mean_absolute_error, explained_variance_score
predictions = model.predict(X_test)
np.sqrt(mean_squared_error(y_test, predictions))

df['price'].describe()

plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, 'r')
plt.show()

single_house = df.drop('price', axis = 1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1, 10))
model.predict(single_house)