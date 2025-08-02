


import os
import random
import numpy as np
import keras

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D



url = 'https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip'
dataset_path = keras.utils.get_file('cats_and_dogs.zip', origin=url, extract=False)

import zipfile
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall('/tmp')


root = '/tmp/PetImages'
print("Arquivos extraídos em:", root)
train_split, val_split = 0.7, 0.15

categories = ['Cat', 'Dog']
category_paths = [os.path.join(root, c) for c in categories]
num_classes = len(categories)

def get_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

n = 100
data = []
for c, category in enumerate(category_paths):
    images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(category) 
              for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    images = images[:n]
    for img_path in images:
        try:
            img, x = get_image(img_path)
            data.append({'x': np.array(x[0]), 'y': c})
        except:
           
            pass

print("Total de imagens carregadas:", len(data))
random.shuffle(data)

idx_val = int(train_split * len(data))
idx_test = int((train_split + val_split) * len(data))

train = data[:idx_val]
val = data[idx_val:idx_test]
test = data[idx_test:]

x_train = np.array([t["x"] for t in train])
y_train = [t["y"] for t in train]

x_val = np.array([t["x"] for t in val])
y_val = [t["y"] for t in val]

x_test = np.array([t["x"] for t in test])
y_test = [t["y"] for t in test]


x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("Train / Val / Test:", len(x_train), len(x_val), len(x_test))
all_images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root) 
              for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]

idx = [int(len(all_images) * random.random()) for i in range(8)]
imgs = [image.load_img(all_images[i], target_size=(224, 224)) for i in idx]
concat_image = np.concatenate([np.asarray(img) for img in imgs], axis=1)

plt.figure(figsize=(16,4))
plt.imshow(concat_image)
plt.axis('off')
plt.show()
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_val, y_val))
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["val_loss"])
ax.set_title("Validation loss")
ax.set_xlabel("Epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history.history["val_accuracy"])
ax2.set_title("Validation accuracy")
ax2.set_xlabel("Epochs")
ax2.set_ylim(0, 1)

plt.show()
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
vgg = keras.applications.VGG16(weights='imagenet', include_top=True)
vgg.summary()


inp = vgg.input
out = Dense(num_classes, activation='softmax')(vgg.layers[-2].output)
model_new = Model(inputs=inp, outputs=out)


for layer in model_new.layers[:-1]:
    layer.trainable = False
model_new.layers[-1].trainable = True

model_new.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_new.summary()
history2 = model_new.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_val, y_val))
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["val_loss"], label='CNN do zero')
ax.plot(history2.history["val_loss"], label='VGG16 Transfer')
ax.set_title("Validation loss")
ax.set_xlabel("Epochs")
ax.legend()

ax2 = fig.add_subplot(122)
ax2.plot(history.history["val_accuracy"], label='CNN do zero')
ax2.plot(history2.history["val_accuracy"], label='VGG16 Transfer')
ax2.set_title("Validation accuracy")
ax2.set_xlabel("Epochs")
ax2.set_ylim(0, 1)
ax2.legend()

plt.show()
loss, accuracy = model_new.evaluate(x_test, y_test, verbose=0)
print('Test loss (VGG16):', loss)
print('Test accuracy (VGG16):', accuracy)
img, x = get_image(all_images[0])
probabilities = model_new.predict(x)
print("Probabilidades:", probabilities)
