import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress FutureWarnings

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress TensorFlow deprecation warnings

import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from transformers import pipeline

# Step 1: Load doctor data from JSON
def load_doctors_from_json(json_filename):
    with open(json_filename, 'r', encoding='utf-8') as jsonfile:
        doctors = json.load(jsonfile)
    # Validate and clean data if necessary
    return [
        {
            "name": doc.get("name", "Unknown"),
            "specialty": doc.get("specialty", "Unknown"),
            "address": doc.get("address", "Unknown"),
            "practice_name": doc.get("practice_name", "Unknown"),
            "overall_rating": doc.get("overall_rating", 0),
            "languages_spoken": doc.get("languages_spoken", []),
            "insurances": doc.get("insurances", []),
            "reviews": doc.get("reviews", []),
            "highlights": doc.get("highlights", {}),
        }
        for doc in doctors
    ]

# Step 2: Initialize the embedding model with a more advanced model
model = SentenceTransformer('all-mpnet-base-v2')

# Normalize embeddings for better similarity calculations
def normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Step 3: Create embeddings for the doctor database
# Embeddings are numerical representations of the doctor descriptions, 
# which allow for efficient similarity calculations. These embeddings 
# are used to compare the patient query with the doctor database to 
# find the most relevant matches based on the context of the query.
def create_doctor_embeddings(doctors_data):
    """
    Turn a list of doctor records (or the full JSON dict) into
    a list of embedding vectors.
    """
    # Normalize input to a list of doctor dicts
    if isinstance(doctors_data, dict) and "doctors" in doctors_data:
        doctors = doctors_data["doctors"]
    else:
        doctors = doctors_data

    print("Creating doctor embeddings...")
    descriptions = []

    for doc in doctors:
        # Skip non-dict entries
        if not isinstance(doc, dict):
            continue

        # Build education entries
        edu_entries = []
        for edu in doc.get("education_and_background", []):
            if isinstance(edu, dict):
                degree = edu.get("degree", "")
                school = edu.get("school", "")
                year = edu.get("year", "")
                edu_entries.append(f"{degree} from {school} ({year})")

        # Construct description string
        desc = (
            f"{doc.get('name', 'Unknown')} is a doctor specializing in "
            f"{', '.join(doc.get('areas_of_expertise', []))}. "
            f"About the doctor: {doc.get('about_doctor', 'No details available')}. "
            f"They are affiliated with {', '.join(doc.get('hospital_affiliations', []))}. "
            f"The office is at {doc.get('office_location', {}).get('address', 'Unknown')}, "
            f"{doc.get('office_location', {}).get('city', 'Unknown')}, "
            f"{doc.get('office_location', {}).get('state', 'Unknown')} "
            f"{doc.get('office_location', {}).get('zip', 'Unknown')}. "
            f"Languages spoken: {', '.join(doc.get('languages_spoken', []))}. "
            f"Accepts: {', '.join(doc.get('in_network_insurance', []))}. "
            f"Overall rating: {doc.get('overall_online_rating', 0)}. "
            f"Education & background: {', '.join(edu_entries)}. "
            f"Availability — today: {doc.get('availability', {}).get('today', 'Unknown')}, "
            f"in two days: {doc.get('availability', {}).get('in_two_days', 'Unknown')}, "
            f"in one week: {doc.get('availability', {}).get('in_one_week', 'Unknown')}, "
            f"in one month: {doc.get('availability', {}).get('in_one_month', 'Unknown')}."
        )

        descriptions.append(desc)

    # Encode and return embeddings
    return model.encode(descriptions)


# Step 4: Build the FAISS index
def build_faiss_index(embeddings):
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# Step 5: Define the RAG function
def find_top_doctors(patient_query, index, doctors, weights=None):
    """
    Finds the top 5 most relevant doctors for a given patient query.
    :param patient_query: Natural language query from the patient.
    :param index: FAISS index containing doctor embeddings.
    :param doctors: List of doctor dictionaries.
    :param weights: Dictionary of weights for attributes (e.g., specialty, location).
    :return: List of top 5 doctors.
    """
    print("Finding top doctors...")
    # Use an LLM to parse and rearrange the patient query into a comparable format

    # Initialize the LLM pipeline (ensure the model is downloaded and available)
    llm = pipeline("text2text-generation", model="google/flan-t5-base")

    # Generate a structured description for the patient query
    prompt = (
        f"Rearrange the following patient query into a structured JSON format comparable to the doctor data format which is"
        f" {doctors[0]}. "
        f"Ensure the output includes fields in the following order: 'areas_of_expertise', 'office_location', "
        f"'in_network_insurance', 'languages_spoken', and any other relevant details. "
        f"Patient query: '{patient_query}'."
    )
    patient_query_description = llm(prompt)[0]['generated_text']

    # Log the rearranged patient query to the console
    print("Rearranged Patient Query:", patient_query_description)

    # Encode and normalize the patient query
    patient_embedding = normalize_embeddings(model.encode([patient_query_description]))

    # Perform similarity search
    distances, indices = index.search(np.array(patient_embedding), k=5)

    # Retrieve the top 5 doctors with distances
    top_doctors = []
    for j, i in enumerate(indices[0]):
        doctor = doctors[i]
        similarity_score = 1 - distances[0][j]

        # Attribute-based scoring (optional)
        attribute_score = 0
        if weights:
            for key, weight in weights.items():
                if key in doctor and key in patient_query:
                    if isinstance(doctor[key], str):
                        attribute_score += weight * fuzz.ratio(doctor[key].lower(), patient_query[key].lower()) / 100
                    elif isinstance(doctor[key], list):
                        attribute_score += weight * max(
                            fuzz.ratio(item.lower(), patient_query[key].lower()) / 100 for item in doctor[key]
                        )

        # Combine similarity and attribute scores
        total_score = similarity_score + attribute_score
        top_doctors.append({"doctor": doctor, "score": round(total_score, 2)})

    # Sort by combined score
    top_doctors = sorted(top_doctors, key=lambda x: x["score"], reverse=True)
    return top_doctors[:5]

# Initialize Flask app
app = Flask(__name__)

# Load doctor data and build FAISS index at startup
json_filename = "doctors_us.json"  # Ensure this file is in the same directory as the script

# Step 1: Load doctor data
with open(json_filename, "r", encoding="utf-8") as f:
    doctors = json.load(f)["doctors"]  # Extract the "doctors" list from the JSON file

# Step 2: Create embeddings
raw_embeddings = create_doctor_embeddings(doctors)

# Step 3: Normalize embeddings for cosine similarity
doctor_embeddings = normalize_embeddings(raw_embeddings)
print(f"Created and normalized embeddings for {len(doctor_embeddings)} doctors.")

# Step 4: Build the FAISS index
index = build_faiss_index(doctor_embeddings)
print("FAISS index built successfully.")

# Flask endpoint
@app.route('/find_doctors', methods=['POST'])
def find_doctors():
    """
    Endpoint to find top 5 doctors based on patient query.
    Expects a JSON payload with 'query' (natural language query).
    Returns a JSON response with the top 5 doctors.
    """
    print("Received request to find doctors...")
    data = request.get_json()
    if 'query' not in data:
        return jsonify({"error": "Missing 'query' field in request"}), 400

    patient_query = data['query']

    # Perform similarity search on all doctors
    patient_embedding = model.encode([patient_query])
    distances, indices = index.search(np.array(patient_embedding), k=5)

    # Rank and format the response
    weights = {"specialty": 0.5, "address": 0.3, "languages_spoken": 0.2}
    top_doctors = find_top_doctors(patient_query, index, doctors, weights=weights)

    return app.response_class(
        response=json.dumps(top_doctors, indent=4),
        mimetype='application/json'
    )

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)