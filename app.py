import streamlit as st
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    stop_words = set(stopwords.words('english'))  # Set of stopwords
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)  # Return cleaned text

# Title and header
st.markdown("<h1 style='text-align: center; color: #4CAF50; font-size: 36px;'>Spam Email Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #333; font-size: 24px;'>Welcome to the Spam Email Detection App</h2>", unsafe_allow_html=True)

# Custom CSS for styling
st.markdown("""
<style>
    body {
        background-color: #e8f5e9; /* Light green background */
        font-family: Arial, sans-serif;
    }
    .stTextInput, .stSelectbox, .stSlider, .stTextArea {
        background-color: #ffffff; 
        border: 2px solid #4CAF50; 
        border-radius: 5px; 
        padding: 10px;
        font-size: 16px; 
    }
    .stTextInput:focus, .stSelectbox:focus, .stSlider:focus, .stTextArea:focus {
        border-color: #388e3c; 
        box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
    }
    .stButton {
        color: #4CAF50; 
        background-color: #ffffff; /* White button background */
        border-radius: 5px; 
        padding: 12px 20px; 
        font-size: 16px;
        border: 2px solid #4CAF50; /* Green border */
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .stButton:hover {
        background-color: #4CAF50; /* Green background on hover */
        color: white; /* White text on hover */
    }
    .highlight {
        background-color: #ffeb3b; 
        padding: 5px; 
        border-radius: 3px; 
        font-weight: bold;
    }
    .styled-dataframe {
        border: 1px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
        background-color: #ffffff; /* White background for DataFrame */
    }
    .styled-dataframe th {
        background-color: #4CAF50; /* Green header */
        color: white; /* White text for header */
    }
    .styled-dataframe td {
        color: #333; /* Dark text for cells */
    }
    .custom-text {
        color: #333;
        font-size: 18px;
        padding: 5px;
        border-radius: 3px;
        background-color: #ffffff; /* Background for text */
        border: 1px solid #4CAF50; /* Border color */
    }
</style>
""", unsafe_allow_html=True)

# Text input for name
name = st.text_input("Enter your name:", key='name_input')
if st.button("Submit"):
    if name:
        st.markdown(f"<div class='custom-text'>Welcome, <span class='highlight'>{name}</span>!</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='custom-text'>Please enter your name.</div>", unsafe_allow_html=True)

# Selectbox for favorite programming language
programming_languages = ["Python", "Java", "JavaScript", "C++", "Ruby"]
favorite_language = st.selectbox("Choose your favorite programming language:", programming_languages, key='language_select')
st.markdown(f"<div class='custom-text'>Your favorite programming language is: <span class='highlight'>{favorite_language}</span></div>", unsafe_allow_html=True)

# Slider to select a number from 1 to 100
number = st.slider("Select a number between 1 and 100:", 1, 100, key='number_slider')
st.markdown(f"<div class='custom-text'>You selected: <span class='highlight'>{number}</span></div>", unsafe_allow_html=True)

# Checkbox to display a message
if st.checkbox("Check this box to display a message"):
    st.markdown("<div class='custom-text'>Thank you for checking the box!</div>", unsafe_allow_html=True)

# Radio button for skill level selection
skill_level = st.radio("Select your skill level:", ("Beginner", "Intermediate", "Advanced"), key='skill_radio')
st.markdown(f"<div class='custom-text'>You selected: <span class='highlight'>{skill_level}</span></div>", unsafe_allow_html=True)

# File uploader for uploading a CSV file
uploaded_file = st.file_uploader("Upload a CSV file of emails:", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("<div class='custom-text'>Here are the contents of your uploaded CSV file:</div>", unsafe_allow_html=True)
    st.dataframe(df)  # Display the original dataframe
    
    # Check if required columns exist
    if 'Message' in df.columns and 'Category' in df.columns:  
        # Data preprocessing
        df['cleaned_text'] = df['Message'].apply(preprocess_text)  # Clean the message column
        st.markdown("<div class='custom-text'>Cleaned Dataset:</div>", unsafe_allow_html=True)
        st.markdown("<div class='styled-dataframe'>", unsafe_allow_html=True)
        st.dataframe(df[['Message', 'cleaned_text']], height=300)  # Display cleaned text
        st.markdown("</div>", unsafe_allow_html=True)

        # Visualization of spam vs. non-spam emails
        spam_count = df['Category'].value_counts()
        plt.bar(spam_count.index, spam_count.values, color='#4CAF50')  # Green color for bars
        plt.title("Distribution of Email Categories")
        plt.xlabel("Category")
        plt.ylabel("Count")
        st.pyplot(plt)

        # Word cloud visualization for spam emails
        spam_emails = ' '.join(df[df['Category'] == 'spam']['cleaned_text'])  # Filter for spam messages
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_emails)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # Train a Naïve Bayes classifier
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['cleaned_text'])
        y = df['Category']  # Use the category column
        model = MultinomialNB()
        model.fit(X, y)

        # Text input for email classification
        email_input = st.text_area("Enter an email to classify:")
        if st.button("Classify"):
            input_vector = vectorizer.transform([preprocess_text(email_input)])
            prediction = model.predict(input_vector)
            st.markdown(f"<div class='custom-text'>This email is classified as: <span class='highlight'>{prediction[0]}</span></div>", unsafe_allow_html=True)  # Displays the predicted category
    else:
        st.error("The uploaded CSV file must contain 'Message' and 'Category' columns.")
