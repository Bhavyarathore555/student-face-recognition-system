## System Workflow

1. Student images are collected and stored.
2. Face encodings are generated using the `face_recognition` library.
3. Encodings are serialized and stored permanently.
4. Webcam captures live video frames using OpenCV.
5. Haar Cascade detects faces in real time.
6. Live face encodings are generated and compared with stored encodings.
7. Matching student names are displayed on the screen.
## Optimization Implemented

To improve the efficiency of the system, face encodings are generated once and stored permanently using serialized storage (`pickle`) instead of generating encodings repeatedly at every program startup.

This optimization significantly reduces startup computation time and improves scalability when handling multiple student records.

### Previous Approach

* Load all student images at startup
* Generate face encodings every time the application runs
* Higher computational overhead

### Optimized Approach

* Generate encodings once
* Store encodings in `encodings.pkl`
* Load stored encodings directly during execution
* Faster initialization and improved performance

This approach is similar to how real-world facial recognition systems manage embeddings efficiently.
