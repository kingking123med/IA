import cv2
import dlib
from scipy.spatial import distance as dist

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define constants for the eye aspect ratio and the number of consecutive frames the eyes must be closed to trigger an alarm
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

# Initialize the frame counter and the total number of blinks
COUNTER = 0
TOTAL = 0

# Initialize the video stream and allow the camera sensor to warm up
vs = cv2.VideoCapture(0)
while True:
    # Get a new frame from the video stream
    ret, frame = vs.read()
    
    # Check if the frame is empty
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    for rect in rects:
        # Determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates and compute the eye aspect ratio
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Check if the eye aspect ratio is below the blink threshold, and increment the blink counter if it is
        if ear < EYE_AR_THRESH:
            COUNTER += 1

        # Otherwise, reset the blink counter
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0

    # Display the current number of blinks on the frame
    cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Frame", frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()
