import os
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = io.imread(img_path)
        img = resize(img, (50, 50))  # resize ke 50 50 
        images.append(img) 
        labels.append(folder.split("/")[-1])
    return np.array(images), np.array(labels)

def load_test_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = io.imread(img_path)
        img = resize(img, (50, 50))
        images.append(img)  
    return np.array(images)

# Load images and labels
car_images, car_labels = load_images_from_folder('./train/car')
bus_images, bus_labels = load_images_from_folder('./train/bus')
truck_images, truck_labels = load_images_from_folder('./train/truck')

# Combine data dari smeua kelas
X = np.concatenate([car_images, bus_images, truck_images], axis=0)
y = np.concatenate([car_labels, bus_labels, truck_labels])

#  KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train 
knn_classifier.fit(X.reshape(X.shape[0], -1), y)  # Flatten the images for KNN

# Load  images
test_images = load_test_images_from_folder('./test')

# predictions  test set
test_predictions = knn_classifier.predict(test_images.reshape(test_images.shape[0], -1))

# predictions dan display  images
for i, (image, prediction) in enumerate(zip(test_images, test_predictions)):
    # Display the image
    plt.imshow(image)
    plt.title(f"Image {i + 1}: {prediction}")
    plt.axis('off')
    plt.show()
