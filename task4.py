import cv2
import mediapipe as mp

# Input video file
input_video_path = "task_4_video.mp4"

# Create VideoCapture object
cap = cv2.VideoCapture(input_video_path)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Adjust the codec accorging to your preference
output_video_path = "output_task_4.mp4"
out = cv2.VideoWriter(output_video_path, fourcc, 30, (640, 480))  # Adjust the resolution as needed

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to RGB for MediaPipe Pose
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(rgb_frame)

    # Draw the skeleton on the frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow('Pose Estimation', frame)

    # Write the frame to the output video
    out.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
