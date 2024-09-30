import streamlit as st
import moviepy.editor as mp
import speech_recognition as sr
import nltk
import g4f
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Ensure NLTK data is downloaded and specify the path
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

# Set up directories
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Streamlit application title
st.title("Video Analysis and Content Generation")

def video_to_audio(video_file, audio_file='audio.wav'):
    try:
        video = mp.VideoFileClip(video_file)
        
        if video.audio is None:
            raise RuntimeError("No audio track found in the video.")

        video.audio.write_audiofile(audio_file)
        return audio_file
    except Exception as e:
        st.error(f"Error converting video to audio: {str(e)}")
        raise RuntimeError("Failed to convert video to audio")


# Function to convert audio to text using Speech-to-Text API
def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        raise RuntimeError("Speech was unintelligible")
    except sr.RequestError as e:
        raise RuntimeError(f"Speech recognition failed: {str(e)}")

# Function to extract keywords using TF-IDF and KMeans
def extract_keywords(text, num_keywords=10):
    try:
        words = nltk.word_tokenize(text.lower())
        words = [word for word in words if word.isalnum()]

        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85)
        X = vectorizer.fit_transform(words)

        num_clusters = 1
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)

        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()

        keywords = []
        for i in range(num_clusters):
            for ind in order_centroids[i, :num_keywords]:
                keywords.append(terms[ind])

        return keywords
    except Exception as e:
        st.error(f"Error extracting keywords: {str(e)}")
        raise RuntimeError("Failed to extract keywords")

# Function to generate advanced content using g4f
def generate_advanced_content_g4f(text, keywords):
    try:
        prompt_titles = f"Generate 5 unique and catchy video titles based on these keywords: {', '.join(keywords)}."
        prompt_description = f"Write a detailed and engaging video description based on the following text: '{text}'."
        prompt_tags = f"Generate a list of 15 tags relevant to the following keywords: {', '.join(keywords)}."

        # Create the messages list required by g4f
        messages_titles = [{"role": "user", "content": prompt_titles}]
        messages_description = [{"role": "user", "content": prompt_description}]
        messages_tags = [{"role": "user", "content": prompt_tags}]

        titles_response = g4f.ChatCompletion.create(messages=messages_titles, model="gpt-3.5-turbo")
        title_suggestions = [title.strip() for title in titles_response.split("\n") if title.strip()]

        description_response = g4f.ChatCompletion.create(messages=messages_description, model="gpt-3.5-turbo")
        detailed_description = description_response

        tags_response = g4f.ChatCompletion.create(messages=messages_tags, model="gpt-3.5-turbo")
        tags_list = [tag.strip() for tag in tags_response.split(",")]

        return title_suggestions, detailed_description, tags_list
    except Exception as e:
        st.error(f"Error generating content: {str(e)}")
        raise RuntimeError("Failed to generate advanced content")

# Function to generate hashtags
def generate_hashtags(tags):
    return " ".join(["#" + tag.replace(" ", "") for tag in tags])

# Upload video file using Streamlit
uploaded_file = st.file_uploader("Upload your video file", type=["mp4", "mov", "avi"])

# Main video processing workflow
if uploaded_file is not None:
    video_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info("Video uploaded successfully.")
    
    with st.spinner("Processing your video..."):
        try:
            # Convert video to audio
            audio_path = video_to_audio(video_path)
            st.success("Audio extracted successfully.")

            # Convert audio to text
            extracted_text = audio_to_text(audio_path)
            st.success("Text transcribed successfully.")
            st.text_area("Transcribed Text", extracted_text, height=200)

            # Extract keywords and generate content
            keywords = extract_keywords(extracted_text)
            title_suggestions, detailed_description, tags_list = generate_advanced_content_g4f(extracted_text, keywords)
            
            # Display results
            st.subheader("Generated Content")
            st.write("**Video Titles:**")
            for i, title in enumerate(title_suggestions, 1):
                st.write(f"{i}. {title}")

            st.write("**Video Description:**")
            st.text_area("Description", detailed_description, height=150)

            st.write("**Tags:**")
            st.text_area("Tags", "\n".join(tags_list), height=100)

            st.write("**Hashtags:**")
            hashtags = generate_hashtags(tags_list)
            st.text_area("Hashtags", hashtags, height=50)
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
        finally:
            # Clean up temporary files
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)
