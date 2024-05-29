import pandas as pd
import os
import glob as gb
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

TRAIN_DIR = "C:/My Folder/IPD/Image/train"
TEST_DIR = "C:/My Folder/IPD/Image/test"
BATCH_SIZE=64

for folder in os.listdir(TRAIN_DIR):
    files = gb.glob(pathname= str(TRAIN_DIR+ '/'+ folder + '/*.jpg'))
    print(f'For training data, found {len(files)} in folder {folder}')

for folder in os.listdir(TEST_DIR):
    files = gb.glob(pathname= str(TEST_DIR+ '/'+ folder + '/*.jpg'))
    print(f'For testing data, found {len(files)} in folder {folder}')

import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def view_random_image(target_dir, target_class):
    # We will view images from here
    target_folder = target_dir + target_class

    # Get a random image path
    random_image = random.sample(os.listdir(target_folder), 1)

    # read in the image and plot it using matplolib
    img = mpimg.imread(target_folder+'/'+random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis('off')
    print(f"Image shape {img.shape}")

    return img

class_names = ['angry','disgust','fear','happy','neutral','sad','surprise']

plt.figure(figsize=(20,10))
for i in range(18):
    plt.subplot(3, 7, i+1)
    class_name = random.choice(class_names)
    img = view_random_image(target_dir="C:/My Folder/IPD/Image/train/", target_class=class_name)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(TRAIN_DIR,
                                                 target_size = (128, 128),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(TEST_DIR,
                                            target_size = (128, 128),
                                            batch_size = BATCH_SIZE,
                                            class_mode = 'categorical')

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(16, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 7, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# model summary
classifier.summary()

history = classifier.fit(training_set,epochs = 20, validation_data = test_set)


classifier.save('model1.h5')  # creates a HDF5 file 'my_model.h5'
classifier.evaluate(test_set)

pd.DataFrame(history.history)[['loss','val_loss']].plot()
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')

model_path = "model1.h5"
loaded_model = keras.models.load_model(model_path)

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

image = cv2.imread("C:/Users/durva/Downloads/smiling.jpg")

image_fromarray = Image.fromarray(image, 'RGB')
resize_image = image_fromarray.resize((128, 128))
expand_input = np.expand_dims(resize_image,axis=0)
input_data = np.array(expand_input)
input_data = input_data/255

pred = loaded_model.predict(input_data)
result = pred.argmax()
result
training_set.class_indices
import pandas as pd
import numpy as np
import cv2
from textblob import TextBlob
import re
from tensorflow import keras
from keras.models import load_model

# Load the pre-trained image classification model
image_model_path = "model1.h5"
image_model = load_model(image_model_path)

# Define function to preprocess the input image
def preprocess_input_image(image):
    image_resized = cv2.resize(image, (128, 128))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    input_data = np.array(image_rgb)
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data / 255.0
    return input_data

# Function to predict image sentiment
def predict_image_sentiment(image):
    input_data = preprocess_input_image(image)
    predictions = image_model.predict(input_data)
    sentiment_score = np.argmax(predictions)
    return sentiment_score

# Load the emoji sentiment mapping
emoji_emotions = {
    '😊': 'positive', '💪': 'positive', '😅': 'positive', '😂': 'positive',
    '🎥': 'positive', '🤩': 'positive', '🎶': 'positive', '😍': 'positive',
    '❤️': 'positive', '🌞': 'positive', '🧺': 'positive', '📚': 'negative',
    '😩': 'negative', '☕️': 'positive', '❄️': 'negative', '👨‍👩‍👧‍👦': 'positive',
    '🙏': 'positive', '✨': 'positive', '💼': 'positive', '🎉': 'positive',
    '🤒': 'negative', '☔️': 'negative', '💻': 'positive', '🍕': 'positive',
    '✈️': 'positive', '🏝️': 'positive', '😆': 'positive', '👍': 'positive',
    '🎨': 'positive', '💖': 'positive', '😢': 'negative', '💔': 'negative',
    '🛀': 'positive', '🌿': 'positive', '🏆': 'positive', '📊': 'negative',
    '🥳': 'positive', '🌈': 'positive', '🎯': 'positive', '🙌': 'positive',
    '🌸': 'positive', '😕': 'negative', '😄': 'positive', '💆‍♀️': 'positive',
    '💅': 'positive', '☮️': 'positive', '🕊️': 'positive', '📖': 'positive',
    '🤗': 'positive', '📸': 'positive', '🥂': 'positive', '🌊': 'neutral',
    '🌱': 'positive', '🎁': 'positive', '😬': 'negative', '🍿': 'positive',
    '🎬': 'positive', '🎵': 'neutral', '💭': 'neutral', '🔮': 'positive',
    '❤️': 'positive', '🤝': 'positive'
}

# Function to extract emojis and emotions from text
def extract_emotions(text):
    emojis = re.findall(r'[^\w\s,]', text)
    emotions = [emoji_emotions.get(e, 'neutral') for e in emojis]
    return emotions

# Function to analyze sentiment of text using TextBlob
def analyze_text_sentiment(text):
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity
    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'

# Function to assign sentiment labels based on extracted emotions
def assign_sentiment(emotions):
    positive_emotions = ['positive', 'excited', 'strong', 'funny', 'love', 'grateful', 'creative', 'relaxed', 'proud', 'accomplished', 'peaceful']
    negative_emotions = ['sad', 'anxious', 'sick']
    has_positive = any(emotion in positive_emotions for emotion in emotions)
    has_negative = any(emotion in negative_emotions for emotion in emotions)
    if has_positive and not has_negative:
        return 'positive'
    elif has_negative and not has_positive:
        return 'negative'
    else:
        return 'neutral'

# Function to perform integrated sentiment analysis
def perform_integrated_sentiment_analysis(input_text, input_image):
    # Extract emotions from text
    text_emotions = extract_emotions(input_text)
    # Analyze sentiment of text
    text_sentiment = analyze_text_sentiment(input_text)
    # Predict sentiment of image
    image_sentiment = predict_image_sentiment(input_image)
    # Assign sentiment based on extracted emotions
    emoji_sentiment = assign_sentiment(text_emotions)
    # Combine sentiment from text, emoji, and image
    combined_sentiment = [text_sentiment, emoji_sentiment, image_sentiment]
    # Determine final sentiment
    final_sentiment = max(set(combined_sentiment), key=combined_sentiment.count)
    return final_sentiment

# Example usage
input_text = "😊 Feeling happy and excited today!"
input_image = cv2.imread("C:/Users/durva/Downloads/smiling.jpg")
result = perform_integrated_sentiment_analysis(input_text, input_image)
print("Integrated Sentiment Analysis Result:", result)



