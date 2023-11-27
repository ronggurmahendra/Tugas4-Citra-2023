import PySimpleGUI as sg
import cv2
import numpy as np
import io
from PIL import Image
import os
# import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
# import numpy as np

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = io.imread(img_path)
        img = resize(img, (50, 50))  # resize ke 50 50 
        images.append(img)  # Keep the image as a 2D array, not flattened
        labels.append(folder.split("/")[-1])  # Use the folder name as the label
    return np.array(images), np.array(labels)

def load_test_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = io.imread(img_path)
        img = resize(img, (50, 50))  # Resize the image to a common size
        images.append(img)  # Keep the image as a 2D array, not flattened
    return np.array(images)

def main():
    # Make the KNN model 
    # Load images and labels knn
    print("  Making KNN Model ...")
    car_images, car_labels = load_images_from_folder('./train/car')
    bus_images, bus_labels = load_images_from_folder('./train/bus')
    truck_images, truck_labels = load_images_from_folder('./train/truck')

    # Combine data from different classes
    X = np.concatenate([car_images, bus_images, truck_images], axis=0)
    y = np.concatenate([car_labels, bus_labels, truck_labels])

    # Create a KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=3)

    # Train the model
    knn_classifier.fit(X.reshape(X.shape[0], -1), y)  # Flatten the images for KNN

    print(" LOADING vehicle_classifier_1.h5 ...")
    # Load Custom Model 1(Transfer learning ResNet50)
    custom_model_1 = load_model('./model/vehicle_classifier_1.h5')

    print(" LOADING vehicle_classifier_2.h5 ...")
    # Load Custom Model 2(Full Tuning ResNet50)
    custom_model_2 = load_model('./model/vehicle_classifier_2.h5')
    

    print(" DOWNLOADING InceptionV3 ...")
    # load Pretrained model InceptionV3
    inceptionV3_model = InceptionV3(weights='imagenet')

    print(" Done Ready to use ...")
    
    # UI Stuff
    # Define the layout
    layout = [
        [sg.Text("Pilih Gambar:")],
        [sg.Input(key="-FILE-", enable_events=True), sg.FileBrowse()],
        [sg.Text("Pilih Metode Classification:")],
        [sg.Drop(values=('Knn (Spek 1)', 'Custom Model 1(Transfer learning ResNet50)', 'Custom Model 2(Full Tuning ResNet50)', 'Pretrained InceptionV3'), default_value='Knn (Spek 1)', key="-OPTION-")],
        [sg.Button("Classify"), sg.Button("Exit")],
        [sg.Image(key="-ORIGINAL-")],
        [sg.Text("Original Image", key="-LABEL-ORIGINAL-", size=(20, 1), font=('Helvetica', 28, 'bold'))]
    ]

    # Create the window
    window = sg.Window("Vehicle classification App Ronggur", layout)

    while True:
        event, values = s.read()

        if event == sg.WINDOW_CLOSED or event == "Exit":
            break

        if event == "Classify":
            image_path = values["-FILE-"]
            clasiffy_option = values["-OPTION-"]

            if image_path:
                # display the iamge
                original_image = cv2.imread(image_path)
                
                # Convert the images to a format that PySimpleGUI can display
                original_imgbytes = cv2.imencode(".png", original_image)[1].tobytes()

                # Update the images on the window
                window["-ORIGINAL-"].update(data=original_imgbytes)

                # predict based off of the option selected
                stringlabel = ''
                stringlabel += "Prediction : "

                print("clasiffy_option : ",clasiffy_option)
                # KNN Prediction
                if clasiffy_option == 'Knn (Spek 1)':

                    img = io.imread(image_path)
                    img = resize(img, (50, 50))  # Resize the image to a common size
                    image_flattened = img.reshape(1, -1)
                    prediction = knn_classifier.predict(image_flattened)
                    print("KNN Predicts : ", prediction)
                    stringlabel += str(prediction[0])

                # custom model 1 prediction 
                elif clasiffy_option == 'Custom Model 1(Transfer learning ResNet50)':

                    img_size = (224, 224)
                    img = image.load_img(image_path, target_size=img_size)
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) 
                    img_array /= 255.0  # Normalize pixel values to between 0 and 1

                    prediction = custom_model_1.predict(img_array)
                    class_index = np.argmax(prediction)

                    class_labels = ['family sedan', 'heavy truck', 'SUV', 'minibus', 'fire engine', 'bus', 'racing car', 'truck', 'jeep', 'taxi']
                    true_label_name = class_labels[class_index]
                    stringlabel += true_label_name

                    print("Custom Model 1 predicts : ", true_label_name)

                elif clasiffy_option == 'Custom Model 2(Full Tuning ResNet50)':
                    
                    img_size = (224, 224)
                    img = image.load_img(image_path, target_size=img_size)
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) 
                    img_array /= 255.0  # Normalize pixel values to between 0 and 1

                    prediction = custom_model_2.predict(img_array)
                    class_index = np.argmax(prediction)

                    class_labels = ['family sedan', 'heavy truck', 'SUV', 'minibus', 'fire engine', 'bus', 'racing car', 'truck', 'jeep', 'taxi']
                    true_label_name = class_labels[class_index]
                    stringlabel += true_label_name

                    print("Custom Model 2 predicts : ", true_label_name)
                    
                elif clasiffy_option == 'Pretrained InceptionV3':

                    img = image.load_img(image_path, target_size=(299, 299))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)

                    # Make predictions
                    predictions = inceptionV3_model.predict(img_array)

                    # Decode and print the top-3 predicted classes
                    decoded_predictions = decode_predictions(predictions, top=1)[0]
                    print("Pretrained InceptionV3 Predictions:")
                    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                        print(f"{i + 1}: {label} ({score:.2f})")
                        stringlabel += label

                window["-LABEL-ORIGINAL-"].update(stringlabel)
    window.close()

if __name__ == "__main__":
    main()
