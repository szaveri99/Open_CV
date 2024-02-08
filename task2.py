import cv2
import numpy as np

# Input video file
input_video_path = "task_2_video.mp4"

# Output video file
output_video_path = "output_video.mp4"

# HSV range values for green color
lower_green = np.array([40, 40, 40])  # Adjust these values for specific green color
upper_green = np.array([80, 255, 255])

# Create VideoCapture object
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Adjust the codec according to your preference
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Function to detect green polka dots and draw red dots
def detect_and_draw(frame):
    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red dots at the center of each green polka dot
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Draw a red dot

    return frame

# Process each frame of the video
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect and draw red dots on green polka dots
    processed_frame = detect_and_draw(frame)

    # Display the processed frame
    cv2.imshow('Output Video', processed_frame)

    # Write the frame to the output video
    out.write(processed_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
