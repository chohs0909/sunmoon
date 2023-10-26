import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

EPOCHS = 300
BATCH_SIZE = 50
IMG_HEIGHT = 150
IMG_WIDTH = 150
optimizer = Adam()

train_dir = "data2/train"
valid_dir = "data2/test"

train_datagen = ImageDataGenerator(rescale=1./255.0,zoom_range=0.2,  # 무작위로 20% 확대 또는 축소
    width_shift_range=0.1,  # 가로 방향으로 10% 이내 이동
    height_shift_range=0.1,  # 세로 방향으로 10% 이내 이동
    brightness_range=[0.8, 1.2])
valid_datagen = ImageDataGenerator(rescale=1./255.0)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(128, 128), batch_size=100, color_mode='grayscale', class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(128, 128), batch_size=100, color_mode='grayscale', class_mode='categorical')

classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu", input_shape=[128, 128, 1]))
classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
classifier.add(tf.keras.layers.Dropout(0.2))
classifier.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
classifier.add(tf.keras.layers.Dropout(0.2))
classifier.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
classifier.add(tf.keras.layers.Dropout(0.2))
classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.Dense(units=256, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=24, activation='softmax'))
classifier.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
classifier.summary()

history = classifier.fit(train_generator, epochs=100, validation_data=valid_generator)

# Save the model
classifier.save("my_model.h5")

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.ylim([0, 4])
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim([0, 1.1])
plt.show()
