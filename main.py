import cv2
import face_recognition
import os

# Load Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Store known face encodings and names
known_face_encodings = []
known_face_names = []

# Path to student images
path = "students"

# Load student images
for filename in os.listdir(path):

    image_path = os.path.join(path, filename)

    # Load image
    image = face_recognition.load_image_file(image_path)

    # Generate face encodings
    encodings = face_recognition.face_encodings(image)

    # Skip image if no face found
    if len(encodings) == 0:
        print(f"No face found in {filename}")
        continue

    # Store first face encoding
    known_face_encodings.append(encodings[0])

    # Store student name without extension
    name = os.path.splitext(filename)[0]
    known_face_names.append(name)

print("Student faces loaded successfully.")

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    # Read webcam frame
    ret, frame = cap.read()

    # If frame not captured properly
    if not ret:
        print("Failed to capture frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Convert frame to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process detected faces
    for (x, y, w, h) in faces:

        # Generate face encoding using detected coordinates
        face_encoding_list = face_recognition.face_encodings(
            rgb_frame,
            [(y, x + w, y + h, x)]
        )
        name = "Unknown"

        # If encoding generated successfully
        if len(face_encoding_list) > 0:

            face_encoding = face_encoding_list[0]

            # Compare with known student faces
            matches = face_recognition.compare_faces(
                known_face_encodings,
                face_encoding
            )

            # If match found
            if True in matches:

                match_index = matches.index(True)

                name = known_face_names[match_index]

        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display student name
        cv2.putText(
            frame,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Display number of faces detected
    cv2.putText(
        frame,
        f'Faces Detected: {len(faces)}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Show webcam output
    cv2.imshow("Student Identification System", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()