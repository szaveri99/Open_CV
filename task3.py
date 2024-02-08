import cv2
import easyocr
import os

# Input folder containing image files
input_folder = "License_Images"
# Output folder to store images with OCR results
output_folder = "License_Outputs"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize the OCR reader using EasyOCR
reader = easyocr.Reader(['en'])

# Function to process each image and add OCR results
def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Perform OCR on the image
    results = reader.readtext(image)

    # Draw red boxes with green-colored OCR text beside them
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Draw a red box around the detected text
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)

        # Draw green-colored OCR text beside the box
        cv2.putText(image, text, (bottom_right[0] + 10, bottom_right[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the processed image to the output folder
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        process_image(image_path)

print("OCR processing completed. Output images saved in 'License_Outputs' folder.")
