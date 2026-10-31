import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
print("Завантаження даних MNIST...")
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
X_test = X_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

print(
    f"Розмірність тренувальної вибірки: {X_train.shape}, Тестової: {X_test.shape}"
)

model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()
print("\nПочаток навчання моделі...")
history = model.fit(
    X_train, y_train, epochs=5, batch_size=64, validation_split=0.1
)
print("\nОцінка моделі на тестових даних...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nТочність на тестових даних (Accuracy): {test_acc*100:.2f}%")
predictions = model.predict(X_test[:5])

plt.figure(figsize=(10, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i]

    color = "green" if predicted_label == true_label else "red"
    plt.title(
        f"Пр: {predicted_label}\nІст: {true_label}",
        color=color,
        fontsize=10,
    )
    plt.axis("off")

plt.tight_layout()
plt.show()