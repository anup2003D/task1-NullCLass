import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Could not load face cascade classifier XML file")

# Load the pre-trained models
age_gender_model = load_model("C:/Users/Anup0/Data Science/Gender and Shirt Colour Detector/Age_Sex_Detection.keras")
shirt_color_model = load_model("C:/Users/Anup0/Data Science/Gender and Shirt Colour Detector/Gender_Shirt_Color_Detection.keras")

# Load the labels for shirt color
label_encoder = LabelEncoder()
label_encoder.fit(['black', 'white', 'other'])

top = tk.Tk()
top.geometry('1000x800')
top.title('Meeting Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background="#CDCDCD", font=('arial', 15, 'bold'))
label2 = Label(top, background="#CDCDCD", font=('arial', 15, 'bold'))
sign_image = Label(top)

def detect_features(image):
    try:
        # Resize the image to match the models' expected input shape
        image_resized = cv2.resize(image, (48, 48))
        
        # Ensure the image has 3 channels (RGB)
        if len(image_resized.shape) == 2:
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
        
        # Normalize the image
        image_normalized = image_resized.astype('float32') / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)

        # Predict age and gender
        age_gender_pred = age_gender_model.predict(image_batch)
        
        # Convert to numpy array if it's a list
        if isinstance(age_gender_pred, list):
            age_gender_pred = np.array(age_gender_pred)

        if age_gender_pred.shape[1] == 1:
            gender = 'Female' if age_gender_pred[0][0] > 0.5 else 'Male'
        elif age_gender_pred.shape[1] == 2:
            gender = 'Female' if age_gender_pred[0][0] > 0.5 else 'Male'
        else:
            print(f"Unexpected output shape from age_gender_model: {age_gender_pred.shape}")
            gender = 'Unknown'

        # Predict shirt color
        shirt_color_pred = shirt_color_model.predict(image_batch)
        shirt_color = label_encoder.inverse_transform([np.argmax(shirt_color_pred[0])])[0]

        # Set age based on shirt color
        if shirt_color == 'white':
            age = 23
        elif shirt_color == 'black':
            age = 'child'
        else:
            age = 'Unknown'

        return gender, age, shirt_color
    except Exception as e:
        print(f"Error in detect_features: {e}")
        import traceback
        traceback.print_exc()
        return 'Error', 'Error', 'Error'
    

def detect_faces(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Convert faces to a list if it's a numpy array, or return an empty list if None
        return faces.tolist() if faces is not None else []
    except Exception as e:
        print(f"Error in detect_faces: {e}")
        import traceback
        traceback.print_exc()
        return []

def Detect(file_path):
    try:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Could not read image from {file_path}")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f"Image shape: {image_rgb.shape}")

        # Detect faces in the image
        faces = detect_faces(image_rgb)

        print(f"Number of faces detected: {len(faces)}")


        if len(faces) == 0:
            print("No faces detected, processing entire image")
            # If no faces are detected, process the entire image
            gender, age, shirt_color = detect_features(image_rgb)
            label = f"{gender}, {age}, {shirt_color}"
            cv2.putText(image_rgb, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            for (x, y, w, h) in faces:
                print(f"Processing face at coordinates: x={x}, y={y}, w={w}, h={h}")
                # Extract the face from the image
                face = image_rgb[y:y+h, x:x+w]
                
                # Detect features
                gender, age, shirt_color = detect_features(face)
                print(f"Detected: Gender={gender}, Age={age}, Shirt Color={shirt_color}")

                # Draw bounding box around the face
                cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

                if age == 'Unknown':
                    label = f"{gender}, {shirt_color}"
                else:
                    label = f"{gender}, {age}"
                
                cv2.putText(image_rgb, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        result_image = Image.fromarray(image_rgb)
        im = ImageTk.PhotoImage(result_image)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(foreground="#011638", text="Detection complete")
        label2.configure(foreground="#011638", text="Check the image for results")

    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()

def show_Detect_Button(file_path):
    Detect_b = Button(top, text="Detect image", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

def Upload_image():
    try:
        file_path = filedialog.askopenfilename(initialdir="C:/Users/Anup0/Data Science/Gender and Shirt Colour Detector/Meeting")
        uploaded = Image.open(file_path)
        
        # Resize the image to a fixed size appropriate for this type of group photo
        new_size = (800, 600)  # This is an estimate, adjust as necessary
        uploaded = uploaded.resize(new_size, Image.LANCZOS)
        
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text=' ')
        label2.configure(text=' ')
        show_Detect_Button(file_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        import traceback
        traceback.print_exc()

#print(f"Image shape after resizing: {image_resized.shape}")
#print(f"Image shape after normalization: {image_normalized.shape}")
#print(f"Image batch shape: {image_batch.shape}")

upload = Button(top, text="Upload an image", command=Upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)

label1.pack(side="bottom", expand=True)
label2.pack(side="bottom", expand=True)
heading = Label(top, text='Age, Gender, and Shirt Color Detector', pady=20, font=("arial", 20, 'bold'))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

top.mainloop()