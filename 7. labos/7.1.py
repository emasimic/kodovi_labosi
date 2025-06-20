import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

(trn_x, trn_y), (tst_x, tst_y) = keras.datasets.mnist.load_data()

for i in range(5, 10):
    plt.imshow(trn_x[i], cmap='gray')
    plt.title(f"Znak: {trn_y[i]}")
    plt.axis('off')
    plt.show()

trn_x = trn_x.astype("float32") / 255
tst_x = tst_x.astype("float32") / 255

trn_x = trn_x.reshape(-1, 784)
tst_x = tst_x.reshape(-1, 784)

trn_y_encoded = keras.utils.to_categorical(trn_y, 10)
tst_y_encoded = keras.utils.to_categorical(tst_y, 10)

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(trn_x, trn_y_encoded, epochs=5, batch_size=32)

train_score = model.evaluate(trn_x, trn_y_encoded, verbose=0)[1]
test_score = model.evaluate(tst_x, tst_y_encoded, verbose=0)[1]
print(f"Preciznost (trening): {train_score:.2f}")
print(f"Preciznost (test): {test_score:.2f}")

test_output = model.predict(tst_x)
test_labels = np.argmax(test_output, axis=1)

cm_test = confusion_matrix(tst_y, test_labels)
ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=range(10)).plot()
plt.title("Test skup")
plt.show()

train_output = model.predict(trn_x)
train_labels = np.argmax(train_output, axis=1)

cm_train = confusion_matrix(trn_y, train_labels)
ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=range(10)).plot()
plt.title("Trening skup")
plt.show()

pogreske = np.where(tst_y != test_labels)[0]
for i in range(5):
    idx = pogreske[i]
    plt.imshow(tst_x[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Točno: {tst_y[idx]}, Pogrešno: {test_labels[idx]}")
    plt.axis('off')
    plt.show()
