import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('TF_Course_Notebooks/DATA/cancer_classification.csv')

df.describe().transpose()

sns.countplot(x='benign_0__mal_1', data=df)
plt.show()

sns.heatmap(df.corr())
plt.show()

X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

X_train.shape

# model = Sequential()
#
# model.add(Dense(30, activation='relu'))
# model.add(Dense(15, activation='relu'))
#
# # BINARY CLASSIFICATION
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimiser='adam')
# model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test))
#
# losses = pd.DataFrame(model.history.history)
# losses.plot()
# plt.show()

model = Sequential()

model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))

# BINARY CLASSIFICATION
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimiser='adam')

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test), callbacks=[early_stop])

losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()


from tensorflow.keras.layers import Dropout
model = Sequential()

model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))

# BINARY CLASSIFICATION
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimiser='adam')
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test), callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

predictions = model.predict_classes(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))