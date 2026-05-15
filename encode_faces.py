import face_recognition
import os
import pickle

# Store encodings and names
known_face_encodings = []
known_face_names = []

# Folder containing student images
path = "students"

# Load images
for filename in os.listdir(path):

    image_path = os.path.join(path, filename)

    # Load image
    image = face_recognition.load_image_file(image_path)

    # Generate encoding
    encodings = face_recognition.face_encodings(image)

    # Skip if no face found
    if len(encodings) == 0:
        print(f"No face found in {filename}")
        continue

    # Store encoding
    known_face_encodings.append(encodings[0])

    # Store name
    name = os.path.splitext(filename)[0]
    known_face_names.append(name)

print("Encodings generated successfully!")

# Save encodings permanently
with open("encodings.pkl", "wb") as file:

    pickle.dump(
        {
            "encodings": known_face_encodings,
            "names": known_face_names
        },
        file
    )

print("Encodings saved successfully!")