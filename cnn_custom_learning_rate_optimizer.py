import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

# Define paths to train, test, and validation directories
train_dir = r"your\train\folder\set"
test_dir = r"your\test\folder\set"
validation_dir = r"your\validation\folder\set"

# Define image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32
epochs = 20

# Generate image data generators for train, test, and validation sets
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Define the learning rates to try
learning_rates = [0.001, 0.005, 0.0001, 0.0005]

# Results dictionaries to store accuracy and loss values for each learning rate
results_accuracy = {}
results_loss = {}
results_val_accuracy = {}
results_val_loss = {}
results_test_accuracy = {}

for lr in learning_rates:
    
    # Define the CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model with custom learning rate
    optimizer = SGD(learning_rate=lr)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        verbose=0)  # Set verbose to 0 to suppress training output
    
    # Evaluate the model on test set
    test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator), verbose=0)
    results_test_accuracy[lr] = test_accuracy
    
    
    # Store the accuracy and loss values for this learning rate
    results_accuracy[lr] = history.history['accuracy']
    results_loss[lr] = history.history['loss']
    results_val_accuracy[lr] = history.history['val_accuracy']
    results_val_loss[lr] = history.history['val_loss']

# Plot the results
plt.figure(figsize=(15, 10))

# Plot Test Accuracy
plt.figure(figsize=(10, 6))
plt.plot(list(results_test_accuracy.keys()), list(results_test_accuracy.values()), marker='o')
plt.title('Test Accuracy vs. Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Test Accuracy')
plt.xscale('log')
plt.grid(True)
plt.show()


# Plot Accuracy
plt.subplot(2, 1, 1)
for lr, acc in results_accuracy.items():
    plt.plot(acc, label='lr={}'.format(lr))
plt.title('LR Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Validation Accuracy
plt.subplot(2, 1, 2)
for lr, val_acc in results_val_accuracy.items():
    plt.plot(val_acc, label='lr={}'.format(lr))
plt.title('LR Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# Plot Loss
plt.subplot(2, 1, 1)
for lr, loss in results_loss.items():
    plt.plot(loss, label='lr={}'.format(lr))
plt.title('LR Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Validation Loss
plt.subplot(2, 1, 2)
for lr, val_loss in results_val_loss.items():
    plt.plot(val_loss, label='lr={}'.format(lr))
plt.title('LR Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()



# Predict classes for test set
Y_pred = model.predict(test_generator, steps=len(test_generator))
y_pred = (Y_pred > 0.5).astype(int)

# Get true classes
y_true = test_generator.classes

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)



# Print classification report
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))


#Getting loss values from training history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'bo', label='Train Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Train and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, Y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

