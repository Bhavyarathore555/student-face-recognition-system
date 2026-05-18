import cv2
import face_recognition
import pickle

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Load saved encodings
with open("encodings.pkl", "rb") as file:
    data = pickle.load(file)

known_face_encodings = data["encodings"]
known_face_names = data["names"]

print("✅ Saved encodings loaded successfully!")

# Start webcam
cap = cv2.VideoCapture(0)

# Frame counter
frame_count = 0

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to capture frame")
        break

    # Increase frame count
    frame_count += 1

    # Skip frames for better performance
    if frame_count % 10 != 0:

        cv2.imshow("Student Identification System", frame)

        # Quit when q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        continue

    # Resize frame to improve speed
    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=0.5,
        fy=0.5
    )

    # Convert to grayscale
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Convert to RGB
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Process each face
    for (x, y, w, h) in faces:

        # Generate encoding
        face_encoding_list = face_recognition.face_encodings(
            rgb_frame,
            [(y, x + w, y + h, x)]
        )

        name = "Unknown"

        # If face encoding found
        if len(face_encoding_list) > 0:

            face_encoding = face_encoding_list[0]

            # Compare with known encodings
            matches = face_recognition.compare_faces(
                known_face_encodings,
                face_encoding
            )

            if True in matches:

                match_index = matches.index(True)

                name = known_face_names[match_index]

        # Scale coordinates back to original frame size
        x1 = x * 2
        y1 = y * 2
        w1 = w * 2
        h1 = h * 2

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x1, y1),
            (x1 + w1, y1 + h1),
            (0, 255, 0),
            2
        )

        # Display name
        cv2.putText(
            frame,
            name,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Display number of faces
    cv2.putText(
        frame,
        f'Faces Detected: {len(faces)}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Show webcam
    cv2.imshow("Student Identification System", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()