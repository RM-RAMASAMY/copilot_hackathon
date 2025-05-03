import csv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Load doctor data from CSV
def load_doctors_from_csv(csv_filename):
    doctors = []
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            doctors.append(row)
    return doctors

# Step 2: Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Create embeddings for the doctor database
def create_doctor_embeddings(doctors):
    doctor_descriptions = [
        f"{doc['Name']} specializes in {doc['Specialty']} located at {doc['Location (zip code)']} "
        f"and accepts {doc['Insurance Provider Supported']}. Speaks {doc['Languages Spoken']}."
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
def find_top_doctors(patient_data, index, doctors):
    """
    Finds the top 5 most relevant doctors for a given patient.
    :param patient_data: Dictionary containing patient details.
    :param index: FAISS index containing doctor embeddings.
    :param doctors: List of doctor dictionaries.
    :return: List of top 5 doctors.
    """
    patient_description = (
        f"Patient {patient_data['name']} born on {patient_data['dob']} "
        f"located at {patient_data['location']} with insurance {patient_data['insurance']} "
        f"has the following health condition: {patient_data['health_condition']}."
    )
    patient_embedding = model.encode([patient_description])
    
    # Filter doctors based on location or insurance
    filtered_doctors = [
        doc for doc in doctors
        if doc['Location (zip code)'] == patient_data['location'] or
        doc['Insurance Provider Supported'] == patient_data['insurance']
    ]
    
    # Perform similarity search
    distances, indices = index.search(np.array(patient_embedding), k=5)
    
    # Retrieve the top 5 doctors with distances
    top_doctors = [{"doctor": doctors[i], "distance": distances[0][j]} for j, i in enumerate(indices[0])]
    return top_doctors

# Step 6: Main script
if __name__ == "__main__":
    # Load doctor data from the CSV file
    csv_filename = 'doctors_data.csv'  # Ensure this file is in the same directory
    doctors = load_doctors_from_csv(csv_filename)
    
    # Create embeddings and build the FAISS index
    doctor_embeddings = create_doctor_embeddings(doctors)
    index = build_faiss_index(doctor_embeddings)
    
    # Example patient data
    patient_data = {
        "name": input("Enter patient name: "),
        "dob": input("Enter patient date of birth (YYYY-MM-DD): "),
        "insurance": input("Enter patient insurance provider: "),
        "location": input("Enter patient ZIP code: "),
        "health_condition": input("Enter patient health condition: ")
    }
    
    # Find the top 5 doctors
    top_doctors = find_top_doctors(patient_data, index, doctors)
    print("Top 5 Doctors for the Patient:")
    for i, entry in enumerate(top_doctors, 1):
        doctor = entry["doctor"]
        distance = entry["distance"]
        print(f"{i}. {doctor['Name']} - {doctor['Specialty']} ({doctor['Location (zip code)']}) - Similarity Score: {1 - distance:.2f}")