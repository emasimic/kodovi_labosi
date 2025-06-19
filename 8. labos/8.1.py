
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), name="conv1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
        layers.Conv2D(64, (3, 3), activation='relu', name="conv2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
        layers.Flatten(name="flatten"),
        layers.Dense(64, activation='relu', name="dense1"),
        layers.Dense(10, activation='softmax', name="output")
    ])
    return model

model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

log_dir = os.path.join("tensorboard_logs")
tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_cb = callbacks.ModelCheckpoint("najbolji_model.h5", save_best_only=True, monitor='val_accuracy')

model.fit(
    x_train, y_train_cat,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[tensorboard_cb, checkpoint_cb]
)

best_model = tf.keras.models.load_model("najbolji_model.h5")

train_loss, train_acc = best_model.evaluate(x_train, y_train_cat, verbose=0)
test_loss, test_acc = best_model.evaluate(x_test, y_test_cat, verbose=0)

print(f"Točnost na skupu za učenje: {train_acc:.4f}")
print(f"Točnost na testnom skupu: {test_acc:.4f}")

y_train_pred = np.argmax(best_model.predict(x_train), axis=1)
y_test_pred = np.argmax(best_model.predict(x_test), axis=1)

conf_train = confusion_matrix(y_train, y_train_pred)
conf_test = confusion_matrix(y_test, y_test_pred)

print("\nMatrica zabune - skup za učenje:")
ConfusionMatrixDisplay(confusion_matrix=conf_train).plot(cmap="Blues")
plt.show()

print("\nMatrica zabune - testni skup:")
ConfusionMatrixDisplay(confusion_matrix=conf_test).plot(cmap="Blues")
plt.show()
