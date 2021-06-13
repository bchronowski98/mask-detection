from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np


def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()


# MODEL``

# Load keras app / pre-trained model

base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet',
                                            input_tensor=tf.keras.layers.Input(shape=(224, 224, 3)))

x = base_model.output
x = tf.keras.layers.AveragePooling2D((7, 7))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

# compile
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4, decay=1e-4 / 20),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# data
image_dir = r'C:\Users\Bartek\Desktop\ML\Projekt\dataset'

images = []
labels = []

for root, folders, filelist in os.walk(image_dir):
    for file in filelist:
        path = os.path.join(root, file)
        print(path)
        img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.applications.xception.preprocess_input(img)
        images.append(img)
        label = path.split('\\')[-2]
        if label == 'mask':
            labels.append(1)
        else:
            labels.append(0)


labels = tf.keras.utils.to_categorical(labels)
images = np.array(images, dtype="float32")

(train_X, test_X, train_Y, test_Y) = train_test_split(images, labels, test_size=0.20, stratify=labels, random_state=42)

datagen_augmented = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
                                                                    zoom_range=0.2,
                                                                    width_shift_range=0.2,
                                                                    height_shift_range=0.2,
                                                                    shear_range=0.2,
                                                                    horizontal_flip=True,
                                                                    fill_mode="nearest")

checkpoint = tf.keras.callbacks.ModelCheckpoint('MaskModel-{epoch:03d}.model', monitor='val_loss', verbose=0,
                                                save_best_only=True, mode='auto')

# Fit the model
history = model.fit(datagen_augmented.flow(train_X, train_Y, batch_size=32),
                    steps_per_epoch=len(train_X) // 32,
                    validation_data=(test_X, test_Y),
                    validation_steps=len(test_X) // 32,
                    epochs=2,
                    callbacks=[checkpoint])

model.save("maskmodel.model", save_format="h5")
plot_loss_curves(history)
plt.savefig("model_plot.png")
print('Model saved')
