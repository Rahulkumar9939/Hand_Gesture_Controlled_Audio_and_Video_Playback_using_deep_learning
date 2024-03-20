import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import os

# Path to your dataset
data_path = r'C:\Users\rahul\Desktop\archive\Collated'

# Define constants
input_shape = (224, 224, 3)
num_classes = 26
batch_size = 32

# Data Augmentation and Normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators for training, validation, and test data
train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    classes=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    classes=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
    subset='validation'
)

# Load MobileNetV2 model pre-trained on ImageNet
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom CNN layers
x = base_model.output
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)

# Add Dense layers for classification
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Compile the model
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001),  # Specify learning rate here
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Fine-tune the model
epochs = 10
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# Ensure the dataset size is sufficient
if steps_per_epoch == 0 or validation_steps == 0:
    print("Dataset size is too small for the specified batch size.")
    print("Please reduce the batch size or increase the dataset size.")
else:
    # Fine-tune layers starting from a specific layer in the base model
    fine_tune_from = 100  # Example: fine-tune starting from layer index 100

    # Unfreeze layers from fine_tune_from onwards
    for layer in base_model.layers[fine_tune_from:]:
        layer.trainable = True

    # Compile the model again after fine-tuning
    model.compile(optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model with fine-tuning
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps
    )

    # Plotting accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(validation_generator)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')

    # Save the trained model
    model.save(r"C:\Users\rahul\Desktop\archive\Hybrid_mobilenetv2_model_finetuned.h5")
