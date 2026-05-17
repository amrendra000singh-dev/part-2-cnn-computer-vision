import pandas as pd   ############part2 -cnn- assignment_module5S#######################

# Load the labels.csv fi
labels_df = pd.read_csv('/content/labels.csv')

# Display the first 5 rows of the DataFrame
display(labels_df.hea



##########################################################################
# Calculate the number of unique classes
num_classes = labels_df['class'].nunique()
print(f"Number of unique classes: {num_classes}")

# Calculate the number of images per class
images_per_class = labels_df['class'].value_counts()
print("\nNumber of images per class:")
print(images_per_class)

# Check for imbalance in the dataset
print("\nChecking for dataset imbalance:")
if images_per_class.min() == images_per_class.max():
    print("The dataset is perfectly balanced.")
else:
    print("The dataset is imbalanced. Here's a breakdown:")
    print(images_per_class.describe())


################################################################################ 


     import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path

# Get unique classes
unique_classes = labels_df['class'].unique()

plt.figure(figsize=(15, 5)) # Adjust figure size as needed

for i, class_name in enumerate(unique_classes):
    # Get the first filename for the current class
    sample_filename_full_path = labels_df[labels_df['class'] == class_name]['filename'].iloc[0]

    # Extract the basename from the filename (e.g., 'normal_001.png')
    image_basename = Path(sample_filename_full_path).name

    # Construct the actual path to the image file in /content/
    image_path = os.path.join('/content/', image_basename)

    # Load the image
    img = Image.open(image_path)

    # Display the image
    plt.subplot(1, len(unique_classes), i + 1)
    plt.imshow(img)
    plt.title(f"Class: {class_name}")
    plt.axis('off')

plt.tight_layout()
plt.show()


################################################################################################


print("\nImage Dimensions per Class:")
for class_name in unique_classes:
    # Get the first filename for the current class
    sample_filename_full_path = labels_df[labels_df['class'] == class_name]['filename'].iloc[0]

    # Extract the basename from the filename (e.g., 'normal_001.png')
    image_basename = Path(sample_filename_full_path).name

    # Construct the actual path to the image file in /content/
    image_path = os.path.join('/content/', image_basename)

    # Load the image and get its dimensions
    img = Image.open(image_path)
    print(f"Class: {class_name}, Dimensions: {img.size} (Width x Height)")

###########################################################################################################
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2 # OpenCV for image processing
from tqdm import tqdm # For progress bar

# Define target image size (based on our exploration, all images are 96x96)
IMG_SIZE = (96, 96)

X = []  # To store image data
y = []  # To store labels

# Group filenames by class to ensure a balanced split
# This is crucial because `labels_df` directly lists paths, but we need to load them.

print("Loading and preprocessing images...")
for index, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
    filename = row['filename']
    class_name = row['class']

    # The dataset has paths like 'images/normal/normal_001.png'.
    # We need to construct the path to the actual file in /content/.
    image_basename = Path(filename).name
    image_path = os.path.join('/content/', image_basename)

    try:
        # Load image using OpenCV
        img = cv2.imread(image_path)
        # Check if the image was loaded successfully
        if img is None:
            print(f"Warning: Could not load image {image_path}. Skipping.")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        img = cv2.resize(img, IMG_SIZE) # Resize to target size
        img = img / 255.0 # Normalize pixel values to [0, 1]
        X.append(img)
        y.append(class_name)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

X = np.array(X) # Convert list of images to numpy array
y = np.array(y) # Convert list of labels to numpy array

print("Preprocessing complete.")
print(f"Shape of image data (X): {X.shape}")
print(f"Shape of labels (y): {y.shape}")

##############################################################################################################


 Encode class labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
# Using stratify to ensure a balanced distribution of classes in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining data shape (X_train): {X_train.shape}")
print(f"Training labels shape (y_train): {y_train.shape}")
print(f"Testing data shape (X_test): {X_test.shape}")
print(f"Testing labels shape (y_test): {y_test.shape}")

print("\nClass distribution in training set:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nClass distribution in testing set:")
print(pd.Series(y_test).value_counts().sort_index())


###################################################################################################################################


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN model architecture
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        # Convolutional Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Convolutional Block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Convolutional Block 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Flattening for Dense Layers
        Flatten(),

        # Dense Layers
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax') # Output layer with softmax for multi-class classification
    ])
    return model

# Get the input shape from our preprocessed data (IMG_SIZE = (96, 96), 3 channels for RGB)
input_shape = (*IMG_SIZE, 3)
# Get the number of unique classes from our dataset (num_classes was already defined as 4)

# Create the model
model = create_cnn_model(input_shape, num_classes)

# Display the model summary
model.summary()


####################################################################

