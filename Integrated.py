import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import cv2
from textblob import TextBlob
import re
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained image classification model
image_model_path = "model1.h5"
image_model = load_model(image_model_path)

# Load the sarcasm detection model
data = pd.read_csv('C:/My Folder/IPD/Sarcasm1.csv')
X_train, X_test, y_train, y_test = train_test_split(data['Sentence'], data[' Label'], test_size=0.2, random_state=42)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)

# Define the emoji sentiment mapping
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

# Function to predict image sentiment
def predict_image_sentiment(image):
    input_data = preprocess_input_image(image)
    predictions = image_model.predict(input_data)
    sentiment_score = np.argmax(predictions)
    return sentiment_score

# Function to preprocess the input image
def preprocess_input_image(image):
    image_resized = cv2.resize(image, (128, 128))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    input_data = np.array(image_rgb)
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data / 255.0
    return input_data

# Function to detect sarcasm in text
def detect_sarcasm(text, tfidf_vectorizer, svm_classifier):
    data = tfidf_vectorizer.transform([text]).toarray()
    output = svm_classifier.predict(data)
    return output

# Function to perform integrated sentiment analysis
def perform_integrated_sentiment_analysis(input_text, input_image, tfidf_vectorizer, svm_classifier):
    # Detect sarcasm in text
    sarcasm_output = detect_sarcasm(input_text, tfidf_vectorizer, svm_classifier)
    if sarcasm_output == 'sarcastic':
        return 'sarcastic'
    else:
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

# Function to display image, text, sentiment, and sarcasm result with full labels
def display_image_text_sentiment_sarcasm(image, text, sentiment, sarcasm):
    # Map sarcasm abbreviation to full label
    sarcasm_label = "Sarcastic" if sarcasm[0].strip() == " S                                " else "Non sarcastic"
    
    plt.figure(figsize=(6, 8))
    # Display image
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.axis('off')
    # Display text with emojis and sarcasm result
    plt.subplot(2, 1, 2)
    text_with_sarcasm = f"{text}\n\nSarcasm: {sarcasm_label}\n\nSentiment: {sentiment}"
    plt.text(0.5, 0.5, text_with_sarcasm, ha='center', va='center', fontsize=12)

    plt.axis('off')
    plt.show()

# Example usage
input_text = "😊 Feeling happy and excited today!"
input_image = cv2.imread("C:/Users/durva/Downloads/smiling.jpg")
sarcasm_result = detect_sarcasm(input_text, tfidf_vectorizer, svm_classifier)
result = perform_integrated_sentiment_analysis(input_text, input_image, tfidf_vectorizer, svm_classifier)
display_image_text_sentiment_sarcasm(input_image, input_text, result, sarcasm_result)


