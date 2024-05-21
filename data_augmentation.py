from keras.preprocessing.image import ImageDataGenerator


# Define path to train directory
train_dir = r"your\train\folder\path"


# Define image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32

# Generate image data generators for train, test, and validation sets
# You can change the values for better results according to your data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.01,
    zoom_range=(0.9, 0.9),
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="constant",
    cval=0)  # Fill the blank pixels with black


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    save_to_dir=r"your\destination\folder\path",  # The path to save the augmented images
    save_prefix='aug',
    save_format='jpg')

# 3200 images are produced at a time
for _ in range(100):  
    train_generator.next()

print("Data augmentation and logging are done.")

