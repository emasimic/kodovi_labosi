import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import datetime

velicina = (48, 48)
paket = 32

train_set = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/Train',
    labels='inferred',
    label_mode='categorical',
    image_size=velicina,
    batch_size=paket,
    validation_split=0.2,
    subset='training',
    seed=123
)

valid_set = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/Train',
    labels='inferred',
    label_mode='categorical',
    image_size=velicina,
    batch_size=paket,
    validation_split=0.2,
    subset='validation',
    seed=123
)

test_set = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/Test',
    labels='inferred',
    label_mode='categorical',
    image_size=velicina,
    batch_size=paket
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

log_path = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
cb_check = ModelCheckpoint('best_model.h5', save_best_only=True)
cb_board = TensorBoard(log_dir=log_path, histogram_freq=1)

model.fit(
    train_set,
    validation_data=valid_set,
    epochs=3,
    callbacks=[cb_check, cb_board]
)

model.load_weights('best_model.h5')
_, tocnost = model.evaluate(test_set)
print(f"Točnost test skupa: {tocnost:.2f}")

real = np.concatenate([lab for _, lab in test_set], axis=0)
probs = model.predict(test_set)
pred = np.argmax(probs, axis=1)
real_lab = np.argmax(real, axis=1)

matrica = confusion_matrix(real_lab, pred)
ConfusionMatrixDisplay(confusion_matrix=matrica).plot(cmap=plt.cm.Blues)
plt.title("Zabuna - test")
plt.show()

try:
    slika = image.load_img('my_sign.jpg', target_size=velicina)
    sl_array = image.img_to_array(slika)
    sl_array = np.expand_dims(sl_array, axis=0) / 255.0
    rez = model.predict(sl_array)
    klasa = np.argmax(rez)
    print(f"Predikcija: {klasa}")
except FileNotFoundError:
    print("Slika nije pronađena.")
