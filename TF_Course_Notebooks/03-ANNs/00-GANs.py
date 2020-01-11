import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])
plt.show()

only_zeros = X_train[y_train ==0]

only_zeros[10]
plt.imshow(only_zeros[14])
plt.show()

import tensorflow as tf
from tensorflow.keras.layers import  Dense, Reshape, Flatten
from tensorflow.keras.models import Sequential

discriminator = Sequential()
discriminator.add(Flatten(input_shape=[28,28]))
discriminator.add(Dense(150, activation = 'relu'))
discriminator.add(Dense(100, activation='relu'))

# FINAL OUTPUT LAYER
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss = 'binary_crossentropy', optimizer='adam')

codings_size = 100
generator = Sequential()
generator.add(Dense(100, activation = 'relu', input_shape=[codings_size]))
generator.add(Dense(150, activation = 'relu'))
generator.add(Dense(784, activation = 'relu'))
generator.add(Reshape([28,28]))

GAN = Sequential([generator, discriminator])
discriminator.trainable = False
GAN.compile(loss = 'binary_crossentropy', optimizer = 'adam')


batch_size = 32
# my_data = X_train
my_data = only_zeros

dataset = tf.data.Dataset.from_tensor_slices((my_data)).shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

epochs = 10
GAN.layers
generator, discriminator = GAN.layers

for epoch in range(epochs):
    print(f'Currently on Epoch {epoch+1}')
    i =0
    for X_batch in dataset:
        i = i+1
        if i%100 ==0:
            print(f'\t Currently on batch number {i} of {len(my_data)//batch_size}')

        #DISCRIMINATOR TRAINING
        noise = tf.random.normal(shape = [batch_size, codings_size])
        gen_images = generator(noise)
        X_fake_vs_real = tf.concat([gen_images, tf.dtypes.cast(X_batch, tf.float32)], axis = 0)

        y1 = tf.constant([[0.0]]*batch_size + [[1.0]]*batch_size)

        discriminator.trainable = True
        discriminator.train_on_batch(X_fake_vs_real, y1)

        #TRAIN GENERATOR
        noise = tf.random.normal(shape=[batch_size, codings_size])
        y2 = tf.constant([[1.0]*batch_size])
        discriminator.trainable = False
        GAN.train_on_batch(noise, y2)

noise = tf.random.normal(shape = [10, codings_size])
noise.shape
plt.imshow(noise)
plt.show()
images = generator(noise)
images.shape
plt.imshow(images[9])
plt.show()
