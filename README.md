# Image Processing Toolkit

This toolkit provides a set of functionalities for image processing, including resizing and selecting regions of interest (ROI), drawing outlines, and overlaying images. The toolkit utilizes the OpenCV and rembg libraries for image manipulation.

## Getting Started

1. **Prerequisites:**
   - Python 3.x
   - OpenCV (`cv2`) library
   - NumPy library
   - `rembg` library

   Install the required libraries using the following command:
   ```bash
   pip install opencv-python numpy rembg
   ```

2. **Usage:**
   - Clone or download this repository.
   - Navigate to the repository directory.

3. **Run the Toolkit:**
   - Execute the `image_processing_toolkit.py` script.
   ```bash
   python image_processing_toolkit.py
   ```

## Toolkit Features

### 1. Resize and Select ROI
   - Use the `resize_and_select_roi` function to resize an image and interactively select a region of interest (ROI).

### 2. Draw Outline
   - Utilize the `draw_outline` function to draw outlines on an image interactively.

### 3. Overlay Images
   - Combine two images by overlaying one onto the other based on keypoint matching.

## Example Usage:

```python
# Example: Resize and Select ROI
cropped_image = resize_and_select_roi("path/to/your/image.jpg")

# Example: Draw Outline
draw_outline("cropped_image.png")

# Example: Overlay Images
background_image_path = "path/to/background/image.jpg"
overlay_image_path = "outlined_image.png"
result_image = overlay_images(background_image_path, overlay_image_path)

# Display the result
cv2.imshow("Overlay Image", result_image)
cv2.imwrite("output_image.png", result_image)

# Remove temporary files
os.remove("outlined_image.png")
os.remove("cropped_image.png")

cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Important Notes:
- Ensure that the required images are present in the specified paths.
- Modify the filenames and paths as needed for your use case.
- To install rembg pacakge it's preferred to have the python interpreter version <3.12

Feel free to explore and customize the toolkit based on your specific image processing requirements.
