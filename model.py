import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import PIL      

# Set the parameters
img_width, img_height = 224, 224
batch_size = 32
num_epochs = 10
num_classes = 8  # Assuming you have 8 classes for the solar system planets

# Preprocess and augment the data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
                
train_generator = train_datagen.flow_from_directory(
    "train",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
)

validation_generator = validation_datagen.flow_from_directory(
    "validation",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
)

# Load the pre-trained MobileNetV2 model, excluding the top layer
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)
)

# Add a global spatial average pooling layer and a fully connected layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(num_classes, activation="softmax")(x)

# Create the transfer learning model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size,
)

# Save the trained model
model.save("planet_identification_model.h5")
