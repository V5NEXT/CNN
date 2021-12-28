
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPool2D
import pickle

#importing dataset from keras
fashion_mnist = tf.keras.datasets.fashion_mnist

#Splitting Dataset into training and test
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# reshape to be [samples][width][height][channels]
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# convert from int to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape)

K = len(set(y_train))
print(f"Number of classes: {K}")


#Intializing the Model
model = tf.keras.models.Sequential()

#adding two convolution Layers, first one acts a input too
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))

#Adding Pooling Layer
model.add(tf.keras.layers.MaxPool2D((2, 2)))

#Adding Flatten Layer to connect between Convalution and Dense Layers
model.add(Flatten())

#First Dense Layer
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))




# for regularization adding droput

model.add(tf.keras.layers.Dropout(0.25))

#L2
# model.add(Dense(10, activation='softmax',kernel_regularizer=regularizers.l2(0.01)))

#L1
# model.add(Dense(10, activation='softmax',kernel_regularizer=regularizers.l1(0.01)))

# Second Dense Layer with softmax
model.add(Dense(10, activation='softmax'))

#SGD Optimizer
# opt = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)


#compiling Model with Adam Optimizer
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# early stopping where val_loss is comapred with loss to prevent over fitting (not used for #3 and #4)
earlystop_callback = tf.keras.callbacks.EarlyStopping(
  monitor='val_loss', min_delta=0.0001,
  patience=1)
# returns model summary
model.summary()


#model with validation set
# r = model.fit(x_train, y_train, validation_data=(x_test, y_test),batch_size=32, epochs=10,callbacks= earlystop_callback)

#Without Validation Set
r = model.fit(x_train, y_train,batch_size=32, epochs=10)

# Plotting
plt.plot(r.history['loss'], label='Loss')
# plt.plot(r.history['val_loss'], label='Validation Loss')
plt.legend()

plt.plot(r.history['accuracy'], label='Accuracy')
# plt.plot(r.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()


#Evaluting random eamples
image_index = 123

plt.imshow(x_test[image_index].reshape(28, 28), cmap='Greys')
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
acc = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(f"Predicted: {acc.argmax()}. Answer: {y_test[image_index]}.")


image_index = 44

plt.imshow(x_test[image_index].reshape(28, 28), cmap='Greys')
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
acc = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(f"Predicted: {acc.argmax()}. Answer: {y_test[image_index]}.")

#Confusion Matrix

# !git clone https://github.com/liady/ssp19ai_utils.git
# !git -C ssp19ai_utils pull

# install above dependecies
import ssp19ai_utils.utils as utils
import importlib
importlib.reload(utils)
predictions = model.predict(x_test)

predicted_classes = utils.label_with_highest_prob(predictions)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Returns drawing of the model Layers
utils.draw_model(model)

# Plot the matrix
utils.plot_confusion_matrix(y_pred=predicted_classes, y_true=y_train, classes=np.array(class_names))

#Question 4

#Checking accuracy with perturbed dataset



# Loading Final Test Set
dictionary = pickle.load(open('fmnist_test_perturb.pickle', 'rb'))

x_perturb, y_perturb = dictionary['x_perturb'], dictionary['y_perturb']
x_perturb = x_perturb.reshape((x_perturb.shape[0], 28, 28, 1))
x_perturb = x_perturb.astype('float32')
x_perturb = x_perturb / 255.0


#Evaluating Model
model.evaluate(x_perturb,y_perturb)


#Data Augmentation
#
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import keras.backend as K
#
#
# def random_reverse(x):
#     if np.random.random() > 0.5:
#         return x[:,::-1]
#     else:
#         return x
#
# def data_generator(X, Y, batch_size=100):
#     while True:
#         idxs = np.random.permutation(len(X))
#         X = X[idxs]
#         Y = Y[idxs]
#         p, q = [], []
#         for i in range(len(X)):
#             p.append(random_reverse(X[i]))
#             q.append(Y[i])
#             if len(p) == batch_size:
#                 yield np.array(p), np.array(q)
#                 p, q = [], []
#         if p:
#             yield np.array(p), np.array(q)
#             p, q = [], []
#
#
# datagen = ImageDataGenerator(
#     featurewise_center=False,  # set input mean to 0 over the dataset
#     samplewise_center=False,  # set each sample mean to 0
#     featurewise_std_normalization=False,  # divide inputs by std of the dataset
#     samplewise_std_normalization=False,  # divide each input by its std
#     zca_whitening=False,  # apply ZCA whitening
#     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#     horizontal_flip=True,  # randomly flip images
#     vertical_flip=False,  # randomly flip images
#     # preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=False
#     )
#
#
# # r = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
# #                                           steps_per_epoch=x_train.shape[0], epochs=8)



# # Loading Final Test Set
# dictionary = pickle.load(open('fmnist_test_perturb.pickle', 'rb'))
#
# x_perturb, y_perturb = dictionary['x_perturb'], dictionary['y_perturb']
# x_perturb = x_perturb.reshape((x_perturb.shape[0], 28, 28, 1))
# x_perturb = x_perturb.astype('float32')
# x_perturb = x_perturb / 255.0
#
#
# #Evaluating Model
# model.evaluate(x_perturb,y_perturb)