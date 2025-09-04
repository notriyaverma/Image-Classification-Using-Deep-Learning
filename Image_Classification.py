# Image Classification using Deep Learning in Google Colab
# This notebook demonstrates image classification with CNNs, data augmentation, and transfer learning

# Install required packages (run this first in Colab)
!pip install tensorflow matplotlib seaborn scikit-learn pillow

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from google.colab import drive, files
import zipfile
import cv2
from PIL import Image
import pandas as pd

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Mount Google Drive (optional - if you want to save models)
# drive.mount('/content/drive')

# =============================================================================
# OPTION 1: Use a built-in dataset (CIFAR-10)
# =============================================================================

def load_cifar10_data():
    """Load and preprocess CIFAR-10 dataset"""
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to categorical
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Number of classes: {num_classes}")

    return (x_train, y_train), (x_test, y_test), class_names

# =============================================================================
# OPTION 2: Upload your own dataset
# =============================================================================

def upload_custom_dataset():
    """Upload and organize custom dataset"""
    print("Upload your dataset as a zip file...")
    print("Expected structure:")
    print("dataset.zip")
    print("├── train/")
    print("│   ├── class1/")
    print("│   │   ├── image1.jpg")
    print("│   │   └── image2.jpg")
    print("│   └── class2/")
    print("│       ├── image1.jpg")
    print("│       └── image2.jpg")
    print("└── test/ (optional)")
    print("    ├── class1/")
    print("    └── class2/")

    uploaded = files.upload()

    # Extract the uploaded zip file
    for filename in uploaded.keys():
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('/content/')
            print(f"Extracted {filename}")

    return '/content/'  # Return base path

# =============================================================================
# Data Augmentation
# =============================================================================

def create_data_generators(train_dir=None, test_dir=None, img_size=(224, 224), batch_size=32):
    """Create data generators with augmentation"""

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2  # Use 20% for validation
    )

    # No augmentation for test data
    test_datagen = ImageDataGenerator(rescale=1./255)

    if train_dir and test_dir:
        # Custom dataset
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )

        val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

        return train_generator, val_generator, test_generator

    return train_datagen, test_datagen

# =============================================================================
# CNN Models
# =============================================================================

def create_simple_cnn(input_shape, num_classes):
    """Create a simple CNN model"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_transfer_learning_model(input_shape, num_classes, base_model_name='VGG16'):
    """Create a transfer learning model"""

    # Load pre-trained base model
    if base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unsupported base model")

    # Freeze base model layers
    base_model.trainable = False

    # Add custom classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# =============================================================================
# Training Function
# =============================================================================

def train_model(model, train_data, val_data, epochs=10, learning_rate=0.001):
    """Train the model"""

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]

    # Train model
    if isinstance(train_data, tuple):
        # CIFAR-10 case
        x_train, y_train = train_data
        x_val, y_val = val_data

        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Custom dataset case
        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )

    return history

# =============================================================================
# Evaluation and Reporting
# =============================================================================

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_data, class_names):
    """Evaluate model and generate reports"""

    if isinstance(test_data, tuple):
        # CIFAR-10 case
        x_test, y_test = test_data
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        predictions = model.predict(x_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
    else:
        # Custom dataset case
        test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
        predictions = model.predict(test_data)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_data.classes

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Create accuracy report dataframe
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()

    return df_report, test_accuracy, cm

def visualize_predictions(model, test_data, class_names, num_samples=9):
    """Visualize model predictions"""

    if isinstance(test_data, tuple):
        x_test, y_test = test_data
        indices = np.random.choice(len(x_test), num_samples, replace=False)
        sample_images = x_test[indices]
        sample_labels = np.argmax(y_test[indices], axis=1)
    else:
        # For generator, get a batch
        sample_images, sample_labels = next(test_data)
        sample_images = sample_images[:num_samples]
        sample_labels = np.argmax(sample_labels[:num_samples], axis=1)

    predictions = model.predict(sample_images)
    predicted_labels = np.argmax(predictions, axis=1)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(sample_images):
            ax.imshow(sample_images[i])
            true_label = class_names[sample_labels[i]]
            pred_label = class_names[predicted_labels[i]]
            confidence = np.max(predictions[i])

            color = 'green' if sample_labels[i] == predicted_labels[i] else 'red'
            ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}',
                        color=color)
            ax.axis('off')

    plt.tight_layout()
    plt.show()

# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("🎯 Image Classification with Deep Learning")
    print("=" * 50)

    # Choose dataset option
    dataset_choice = input("Choose dataset option (1: CIFAR-10, 2: Upload custom): ")

    if dataset_choice == "1":
        print("\n📊 Loading CIFAR-10 dataset...")
        (x_train, y_train), (x_test, y_test), class_names = load_cifar10_data()

        # Split validation data
        split_idx = int(0.8 * len(x_train))
        x_val, y_val = x_train[split_idx:], y_train[split_idx:]
        x_train, y_train = x_train[:split_idx], y_train[:split_idx]

        input_shape = x_train.shape[1:]
        num_classes = len(class_names)

        train_data = (x_train, y_train)
        val_data = (x_val, y_val)
        test_data = (x_test, y_test)

    else:
        print("\n📁 Setting up custom dataset...")
        base_path = upload_custom_dataset()

        # You'll need to modify these paths based on your dataset structure
        train_dir = input("Enter training directory path (e.g., /content/dataset/train): ")
        test_dir = input("Enter test directory path (e.g., /content/dataset/test): ")

        train_data, val_data, test_data = create_data_generators(train_dir, test_dir)

        class_names = list(train_data.class_indices.keys())
        num_classes = len(class_names)
        input_shape = (224, 224, 3)  # Standard for transfer learning

    print(f"\n📝 Dataset Info:")
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Input shape: {input_shape}")

    # Choose model type
    model_choice = input("\nChoose model type (1: Simple CNN, 2: Transfer Learning): ")

    if model_choice == "1":
        print("\n🔧 Creating Simple CNN...")
        model = create_simple_cnn(input_shape, num_classes)
    else:
        base_model_name = input("Choose base model (VGG16/ResNet50): ").upper()
        if base_model_name not in ['VGG16', 'RESNET50']:
            base_model_name = 'VGG16'

        print(f"\n🔧 Creating Transfer Learning model with {base_model_name}...")
        model = create_transfer_learning_model(input_shape, num_classes, base_model_name)

    # Display model summary
    model.summary()

    # Train model
    print("\n🚀 Starting training...")
    epochs = int(input("Enter number of epochs (default 10): ") or "10")
    history = train_model(model, train_data, val_data, epochs=epochs)

    # Plot training history
    print("\n📈 Training History:")
    plot_training_history(history)

    # Evaluate model
    print("\n📊 Evaluating model...")
    df_report, test_accuracy, cm = evaluate_model(model, test_data, class_names)

    # Display accuracy report
    print("\n📋 Detailed Accuracy Report:")
    print(df_report)

    # Visualize predictions
    print("\n🔍 Sample Predictions:")
    visualize_predictions(model, test_data, class_names)

    # Save model
    save_model = input("\nSave model? (y/n): ").lower() == 'y'
    if save_model:
        model_name = input("Enter model name (default: image_classifier): ") or "image_classifier"
        model.save(f'{model_name}.h5')
        print(f"Model saved as {model_name}.h5")

    print(f"\n✅ Training Complete!")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

    return model, history, df_report

# Run the main function
if __name__ == "__main__":
    model, history, accuracy_report = main()

# =============================================================================
# Additional Utility Functions
# =============================================================================

def predict_single_image(model, image_path, class_names, input_size=(224, 224)):
    """Predict a single image"""
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize(input_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")

    # Display image with prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f'Predicted: {class_names[predicted_class]} ({confidence:.2f})')
    plt.axis('off')
    plt.show()

    return class_names[predicted_class], confidence

def create_prediction_pipeline():
    """Create a prediction pipeline for new images"""
    print("🔮 Image Prediction Pipeline")
    print("Upload an image to classify:")

    uploaded = files.upload()

    for filename in uploaded.keys():
        print(f"Predicting for {filename}...")
        predicted_class, confidence = predict_single_image(model, filename, class_names)
        print(f"Result: {predicted_class} with {confidence:.2%} confidence\n")

# Uncomment to run prediction pipeline
# create_prediction_pipeline()