# Doctor Recommendation System

## Overview
The **Doctor Recommendation System** is a machine learning-powered application designed to help patients find the most relevant doctors based on their natural language queries. It uses advanced NLP models, similarity search, and attribute-based scoring to match patient queries with doctor profiles stored in a database. The system is exposed as a REST API built using Flask, making it easy to integrate with other applications.

---

## Features
- **Natural Language Query Processing**: Accepts unstructured patient queries in plain English and converts them into a structured format.
- **Doctor Matching**: Uses embeddings and similarity search to find the most relevant doctors.
- **Attribute-Based Scoring**: Enhances matching by scoring specific attributes like specialty, location, and languages spoken.
- **REST API**: Provides an endpoint to accept patient queries and return the top 5 doctor recommendations.

---

## Built With
### **Programming Language**
- **Python**: The primary language for implementing the backend logic, machine learning, and API.

### **Frameworks and Libraries**
1. **Flask**: For creating the REST API.
2. **Sentence Transformers**: For generating embeddings using the `all-mpnet-base-v2` model.
3. **FAISS**: For efficient similarity search and clustering of dense vectors.
4. **Transformers**: For query restructuring using the `google/flan-t5-base` model.
5. **FuzzyWuzzy**: For flexible attribute-based scoring using fuzzy string matching.
6. **NumPy**: For numerical operations and normalization of embeddings.
7. **TensorFlow**: For suppressing TensorFlow-related warnings.

### **Data Handling**
- **JSON**: Doctor data is stored in a JSON file (`doctors_us.json`), containing detailed information about each doctor.

---

## How It Works
1. **Data Loading**:
   - Doctor data is loaded from a JSON file and validated for consistency.

2. **Embedding Creation**:
   - Doctor profiles are converted into descriptive text strings and encoded into embeddings using the `all-mpnet-base-v2` model.

3. **FAISS Index Construction**:
   - A FAISS index is built using the embeddings for efficient similarity searches.

4. **Query Processing**:
   - Patient queries are processed using the `google/flan-t5-base` model to generate a structured description.
   - The query is encoded into an embedding and normalized.

5. **Similarity Search**:
   - The patient query embedding is compared against the FAISS index to retrieve the top 5 closest matches.

6. **Attribute-Based Scoring**:
   - Additional scoring is applied based on specific attributes like specialty, location, and languages spoken.

7. **Result Formatting**:
   - The top 5 doctors are sorted by their combined similarity and attribute scores and returned as a JSON response.

---

## Challenges Addressed
- Processing unstructured patient queries using NLP models.
- Efficiently searching large datasets with FAISS.
- Flexibly matching attributes with fuzzy string scoring.

---

## API Details
### **Endpoint**: `/find_doctors`
- **Method**: POST
- **Request Payload**:
  ```json
  {
      "query": "I need a cardiologist in New York who speaks Spanish and accepts Blue Cross insurance."
  }
  ```
- **Response**:
  - Returns a list of the top 5 doctors with their details and scores.
  ```json
  [
      {
          "doctor": {
              "name": "Dr. John Doe",
              "specialty": "Cardiology",
              "address": "123 Main St, New York, NY",
              "languages_spoken": ["English", "Spanish"],
              "insurances": ["Blue Cross", "Aetna"],
              "overall_rating": 4.8
          },
          "score": 0.95
      },
      ...
  ]
  ```

---

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the `doctors_us.json` file in the project directory.

4. Run the Flask app:
   ```bash
   python rag_doctor_match.py
   ```

5. Access the API at `http://127.0.0.1:5000/find_doctors`.

---

## Future Enhancements
- **Scalability**: Move the doctor database to a cloud-based solution (e.g., AWS DynamoDB).
- **Real-Time Updates**: Allow real-time updates to the doctor database and embeddings.
- **User Feedback**: Incorporate patient feedback to improve recommendations.
- **Advanced Query Understanding**: Use more advanced models for better query restructuring.


