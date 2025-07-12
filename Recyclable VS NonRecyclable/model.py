import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Set up data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    "dataset", target_size=(224, 224), subset="training", class_mode='categorical'
)
val = datagen.flow_from_directory(
    "dataset", target_size=(224, 224), subset="validation", class_mode='categorical'
)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(7, activation='softmax')  # 7 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train, validation_data=val, epochs=5)

# Save model
model.save("recycle_classifier.h5")

# Save class indices
with open("class_indices.json", "w") as f:
    json.dump(train.class_indices, f)
