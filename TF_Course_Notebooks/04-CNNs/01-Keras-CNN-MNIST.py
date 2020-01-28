import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline

from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train.shape

single_image = x_train[0]

plt.imshow(single_image)
plt.show()

from tensorflow.keras.utils import  to_categorical
y_train.shape

y_example = to_categorical(y_train)
y_example.shape

y_cat_test = to_categorical(y_test, 10)
y_cat_train = to_categorical(y_train, 10)

x_train = x_train/255
x_test = x_test/255

scaled_image = x_train[0]
scaled_image.max()

#batch size, width, height, colot_channels
x_train = x_train.reshape(60000,28, 28, 1)
x_test = x_test.reshape(10000,28, 28, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,MaxPool2D,Flatten

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4,4), strides=(1,1),
                 input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(128, activation='relu'))

# OUTPUT LAYERR
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience = 1)

model.fit(x_train, y_cat_train, epochs=10,
          validation_data=(x_test, y_cat_test), callbacks=[early_stop])

metrics = pd.DataFrame(model.history.history)
metrics[['loss','val_loss']].plot()
plt.show()

metrics[['accuracy','val_accuracy']].plot()
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
predictions= model.predict_classes(x_test)
y_cat_test.shape
y_test
print(classification_report(y_test, predictions))

import seaborn as sns
sns.heatmap(confusion_matrix(y_test, predictions), annot=True)
plt.show()

my_number = x_test[10]
plt.imshow(my_number.reshape(28,28))
plt.show()
model.predict_classes(my_number.reshape(1,28,28,1))


result = model.predict(x_test[17].reshape(1,28,28,1))
print(np.argmax(result))
plt.imshow(x_test[17].reshape(28,28))
plt.show()
