from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


train_dir = r"your\train\folder\path"
test_dir = r"your\test\folder\path"
validation_dir = r"your\validation\folder\path"

# Define image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 64

# Generate image data generators for train, test, and validation sets
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary')


# Define the ResNet50 model
def create_resnet_model(input_shape):
    resnet_model = ResNet50(include_top=False, input_shape=input_shape, weights=None)
    
    model = models.Sequential()
    model.add(resnet_model)
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.GlobalAveragePooling2D())
    
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(layers.Dropout(0.5))
    
    # Change num_classes to 1 for binary classification
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

input_shape = (224, 224, 1)  # Width, Height, Number of channels (3 for RGB)
num_classes = 1  # Change num_classes to 1 for binary classification
resnet_model = create_resnet_model(input_shape)

# Adding early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Optimization algorithm specifies loss function, and evaluation metrics.
resnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Validation_data is used to measure performance during training.
history = resnet_model.fit(train_generator, epochs=25, validation_data=validation_generator, callbacks=[early_stopping])

# Test_data is used to measure performance after training is finished.
test_loss, test_acc = resnet_model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")




# Predict classes for test set
Y_pred = resnet_model.predict(test_generator, steps=len(test_generator))
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
