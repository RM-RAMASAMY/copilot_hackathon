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

# Step 1: Load doctor data from JSON
def load_doctors_from_json(json_filename):
    with open(json_filename, 'r', encoding='utf-8') as jsonfile:
        doctors = json.load(jsonfile)
    return doctors

# Step 2: Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Create embeddings for the doctor database
def create_doctor_embeddings(doctors):
    doctor_descriptions = [
        f"{doc['specialty']} specialist with research experience in {', '.join(doc.get('research_papers', []))}. "
        f"Located at {doc['location']}, speaks {', '.join(doc['languages_spoken'])}, and accepts {', '.join(doc['insurance_providers_accepted'])}."
        for doc in doctors
    ]
    return model.encode(doctor_descriptions)

# Step 4: Build the FAISS index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# Step 5: Define the RAG function
def find_top_doctors(patient_query, index, doctors):
    """
    Finds the top 5 most relevant doctors for a given patient query.
    :param patient_query: Natural language query from the patient.
    :param index: FAISS index containing doctor embeddings.
    :param doctors: List of doctor dictionaries.
    :return: List of top 5 doctors.
    """
    patient_embedding = model.encode([patient_query])

    # Perform similarity search
    distances, indices = index.search(np.array(patient_embedding), k=5)

    # Retrieve the top 5 doctors with distances
    top_doctors = [{"doctor": doctors[i], "distance": distances[0][j]} for j, i in enumerate(indices[0])]
    return top_doctors

# Initialize Flask app
app = Flask(__name__)

# Load doctor data and create FAISS index at startup
json_filename = 'updated_doctors_data.json'  # Ensure this file is in the same directory
doctors = load_doctors_from_json(json_filename)
doctor_embeddings = create_doctor_embeddings(doctors)
index = build_faiss_index(doctor_embeddings)

@app.route('/find_doctors', methods=['POST'])
def find_doctors():
    """
    Endpoint to find top 5 doctors based on patient query.
    Expects a JSON payload with a 'query' field.
    Returns a JSON response with the top 5 doctors.
    """
    data = request.get_json()
    if 'query' not in data:
        return jsonify({"error": "Missing 'query' field in request"}), 400

    patient_query = data['query']
    top_doctors = find_top_doctors(patient_query, index, doctors)

    # Format the response to include all doctor information
    response = [
        {
            "doctor": entry["doctor"],
            "similarity_score": round(1 - entry["distance"], 2)
        }
        for entry in top_doctors
    ]

    return app.response_class(
        response=json.dumps(response, indent=4),
        mimetype='application/json'
    )

if __name__ == "__main__":
    app.run(debug=True)