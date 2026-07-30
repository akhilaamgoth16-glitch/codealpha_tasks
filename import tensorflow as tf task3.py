import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape for CNN
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(train_images, train_labels, epochs=5, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test Accuracy:', test_acc)

# Predict one image
prediction = model.predict(test_images[:1])
print('Predicted digit:', prediction.argmax())

# Display image
plt.imshow(test_images[0].reshape(28,28), cmap='gray')
plt.title(f'Predicted: {prediction.argmax()}')
plt.axis('off')
plt.show()