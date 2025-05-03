import csv
import random

# --- Data Generation Logic (from previous example) ---
# Define possible values for each attribute
specialties = ["Cardiology", "Dermatology", "Pediatrics", "Neurology", "Oncology", "Psychiatry", "General Practice", "Orthopedics", "Gastroenterology", "Pulmonology", "Endocrinology", "Family Medicine", "Internal Medicine", "Ophthalmology", "Otolaryngology (ENT)", "Urology", "Radiology", "Anesthesiology", "Obstetrics & Gynecology (OB/GYN)", "Surgery"]
insurance_options = ["In-Network", "Out-of-Network"]
appointment_options = ["Online", "In-Person"]
# Generate a list of plausible zip codes (using a smaller set for brevity here)
zip_codes = ["90210", "94110", "60611", "10021", "77030", "98104", "02114", "94305", "80206", "33136", "21287", "55455", "92697", "48109", "19104", "63110", "90095", "44106", "27710", "94043"] # Add many more for 1000 unique records
availability_options = ["ASAP", "Flexible"]
level_options = ["High", "Medium", "Low"]
gender_options = ["Male", "Female"]
other_languages = ["Spanish", "Mandarin", "French", "German", "Hindi", "Arabic", "Russian", "Portuguese", "Japanese", "Korean"]
# Generate a list of plausible hospital names (using a smaller set for brevity here)
hospital_affiliations = ["Cedars-Sinai Medical Center", "UCSF Benioff Children's Hospital", "Northwestern Memorial Hospital", "NewYork-Presbyterian/Weill Cornell", "MD Anderson Cancer Center", "Harborview Medical Center", "Massachusetts General Hospital", "Stanford Health Care", "Denver Health Medical Center", "Jackson Memorial Hospital", "Johns Hopkins Hospital", "M Health Fairview University of Minnesota Medical Center", "UCI Medical Center", "University of Michigan Hospitals - Michigan Medicine", "Hospital of the University of Pennsylvania", "Barnes-Jewish Hospital", "UCLA Medical Center", "Cleveland Clinic Main Campus", "Duke University Hospital", "El Camino Hospital"] # Add many more

# Sample first and last names
first_names = ["John", "Jane", "Alex", "Emily", "Chris", "Katie", "Michael", "Sarah", "David", "Laura"]
last_names = ["Smith", "Johnson", "Brown", "Williams", "Jones", "Garcia", "Miller", "Davis", "Martinez", "Hernandez"]

doctors_data = []
num_records = 1000 # Set the number of records to generate

print(f"Generating {num_records} doctor records...")

for i in range(1, num_records + 1):
    doctor = {}
    doctor['id'] = i
    doctor['Name'] = f"{random.choice(first_names)} {random.choice(last_names)}"
    doctor['Specialty'] = random.choice(specialties)
    doctor['insurance provider'] = random.choice(insurance_options)
    doctor['Appointment'] = random.choice(appointment_options)
    doctor['Location (zip code)'] = random.choice(zip_codes)
    doctor['Latest Availability'] = random.choice(availability_options)
    doctor['Experience Level'] = random.choice(level_options)
    doctor['Education'] = random.choice(level_options)
    doctor['Online reviews'] = random.choice(level_options)
    doctor['Gender of Doctor'] = random.choice(gender_options)

    # Languages
    languages_spoken = ["English"]
    if random.random() < 0.4:  # ~40% chance of having more than one language
        num_extra_languages = random.randint(1, min(2, len(other_languages)))
        available_langs = [lang for lang in other_languages if lang not in languages_spoken]
        if available_langs:
            extra_langs = random.sample(available_langs, min(num_extra_languages, len(available_langs)))
            languages_spoken.extend(extra_langs)
    doctor['Languages Spoken'] = ", ".join(languages_spoken)

    # Replace hospital affiliation with insurance provider supported
    doctor['Insurance Provider Supported'] = random.choice(insurance_options)

    doctors_data.append(doctor)

print("Data generation complete.")

# --- CSV Writing Logic ---

# Define the output filename
csv_filename = 'doctors_data.csv'

# Define the header fields (must match the keys in the dictionaries)
# Ensure the order is as desired in the CSV
fieldnames = [
    'id',
    'Name',
    'Specialty',
    'insurance provider',
    'Appointment',
    'Location (zip code)',
    'Latest Availability',
    'Experience Level',
    'Education',
    'Online reviews',
    'Gender of Doctor',
    'Languages Spoken',
    'Insurance Provider Supported'  # Updated field
]

print(f"Writing data to {csv_filename}...")

try:
    # Open the file in write mode ('w') with newline='' to prevent extra blank rows
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        # Create a DictWriter object
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write the data rows from the list of dictionaries
        writer.writerows(doctors_data)

    print(f"Successfully wrote {len(doctors_data)} records to {csv_filename}")

except IOError:
    print(f"Error: Could not write to file {csv_filename}. Check permissions.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

