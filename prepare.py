import os
import shutil

# Set the path to your dataset folder
dataset_folder = 'processed_ds'

# Set the path to create train and validation folders
train_folder = 'train'
validation_folder = 'validation'

# Set the train-validation split ratio
split_ratio = 0.8  # 80% for training, 20% for validation

# Get the list of subfolders (classes) in the dataset folder
classes = os.listdir(dataset_folder)

# Create train and validation folders if they don't exist
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(validation_folder):
    os.makedirs(validation_folder)

# Iterate over each class folder
for class_name in classes:
    class_folder = os.path.join(dataset_folder, class_name)

    # Create train and validation subfolders for each class
    train_class_folder = os.path.join(train_folder, class_name)
    validation_class_folder = os.path.join(validation_folder, class_name)

    if not os.path.exists(train_class_folder):
        os.makedirs(train_class_folder)
    if not os.path.exists(validation_class_folder):
        os.makedirs(validation_class_folder)

    # Get the list of image files in the class folder
    images = os.listdir(class_folder)
    num_images = len(images)

    # Split the images into train and validation sets
    num_train = int(split_ratio * num_images)
    train_images = images[:num_train]
    validation_images = images[num_train:]

    # Copy train images from main folder to train class folder
    for train_img in train_images:
        src_path = os.path.join(class_folder, train_img)
        dst_path = os.path.join(train_class_folder, train_img)
        shutil.copy(src_path, dst_path)

    # Copy validation images from main folder to validation class folder
    for validation_img in validation_images:
        src_path = os.path.join(class_folder, validation_img)
        dst_path = os.path.join(validation_class_folder, validation_img)
        shutil.copy(src_path, dst_path)