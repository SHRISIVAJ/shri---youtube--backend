from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS
import moviepy.editor as mp
import speech_recognition as sr
import nltk
import g4f
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the entire application

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Function to convert video to audio
def video_to_audio(video_file, audio_file='audio.wav'):
    video = mp.VideoFileClip(video_file)
    video.audio.write_audiofile(audio_file)
    return audio_file

# Function to convert audio to text using a Speech-to-Text API
def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
    return text

# Function to extract keywords using TF-IDF and KMeans
def extract_keywords(text, num_keywords=10):
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalnum()]

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85)
    X = vectorizer.fit_transform(words)

    # Using KMeans clustering to identify important terms
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

# Function to generate advanced content including video description using g4f
def generate_advanced_content_g4f(text, keywords):
    # Prepare prompts for title suggestions, description, and tags
    prompt_titles = f"Generate 5 unique and catchy video titles based on these keywords: {', '.join(keywords)}. The titles should be concise and relevant to this text: '{text}' without any unnecessary numbering."
    prompt_description = f"Write a detailed and engaging video description based on the following extracted text: '{text}'. Include the main topics like {', '.join(keywords)}."
    prompt_tags = f"Generate a list of 15 tags relevant to the following keywords: {', '.join(keywords)} and the text: '{text}'. These tags should be SEO-friendly and capture the essence of the content without unnecessary numbering."

    # Create the messages list required by g4f
    messages_titles = [{"role": "user", "content": prompt_titles}]
    messages_description = [{"role": "user", "content": prompt_description}]
    messages_tags = [{"role": "user", "content": prompt_tags}]

    # Generate the title suggestions
    titles_response = g4f.ChatCompletion.create(messages=messages_titles, model="gpt-3.5-turbo")
    title_suggestions = [title.strip() for title in titles_response.split("\n") if title.strip()]

    # Generate the detailed description
    description_response = g4f.ChatCompletion.create(messages=messages_description, model="gpt-3.5-turbo")
    detailed_description = description_response

    # Generate the tags
    tags_response = g4f.ChatCompletion.create(messages=messages_tags, model="gpt-3.5-turbo")
    tags_list = [tag.strip() for tag in tags_response.split(",")]

    return title_suggestions, detailed_description, tags_list

# Function to generate hashtags from tags list correctly formatted
def generate_hashtags(tags):
    # Create a list of hashtags by removing spaces and adding the '#' prefix
    hashtags = ["#" + tag.replace(" ", "") for tag in tags]
    # Join the list of hashtags into a single string separated by a space
    return " ".join(hashtags)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze-video', methods=['POST'])
def analyze_video():
    if 'videoFile' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    
    video_file = request.files['videoFile']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    # Process the video
    audio_path = video_to_audio(video_path)
    extracted_text = audio_to_text(audio_path)
    keywords = extract_keywords(extracted_text)
    title_suggestions, detailed_description, tags_list = generate_advanced_content_g4f(extracted_text, keywords)
    
    # Generate hashtags in the required format
    hashtags = generate_hashtags(tags_list)

    # Format the tags list for line-by-line display with numbers
    formatted_tags_list = "\n".join([f"{idx + 1}. {tag}" for idx, tag in enumerate(tags_list)])  # Numbered line by line

    return jsonify({
        "extractedText": extracted_text,
        "keywords": keywords,
        "titleSuggestions": title_suggestions,
        "videoDescription": detailed_description,
        "tags": formatted_tags_list,  # Send formatted tags as a single string
        "hashtags": hashtags  # Hashtags in the required space-separated format
    })

# Ensure that the app binds to 0.0.0.0 and the correct port
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
