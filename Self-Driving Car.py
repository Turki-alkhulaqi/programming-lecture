import cv2
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import serial

# Open serial port (adjust port as necessary)
ser = serial.Serial('COM5', 9600)  # replace 'COM3' with the port where your Arduino is connected

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tf.keras.models.load_model("keras_model.h5")

# Load the labels
with open('labels.txt','r') as f:
    class_names = f.read().split('\n')

# Start video capture
cap = cv2.VideoCapture(0)  # 0 is usually the built-in webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the image from OpenCV BGR format to PIL RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)

    # Resize the image to be at least 224x224 and then crop from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Convert numpy.int64 to string, then encode to bytes and send command to Arduino
    command = str(index)
    ser.write((command + '\n').encode())

    # Display the resulting frame
    cv2.imshow('frame', frame)
    print("Class:", class_name)
    print("Confidence Score:", confidence_score)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
ser.close()
