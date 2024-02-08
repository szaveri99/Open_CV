import cv2
import numpy as np
from rembg import remove
import os

def resize_and_select_roi(image_path):
    # Read the image
    img = cv2.imread(image_path, 1)

    # Resize the image for interactive selection
    resized_img = cv2.resize(img, (800, 600))

    # Create a window for selecting a region of interest (ROI)
    cv2.namedWindow("Select the area", cv2.WINDOW_NORMAL)

    # Select a region of interest (ROI) in the resized image
    r = cv2.selectROI("Select the area", resized_img, fromCenter=False)

    # Map the selected region coordinates back to the original image
    scale_factor_x = img.shape[1] / resized_img.shape[1]
    scale_factor_y = img.shape[0] / resized_img.shape[0]

    original_roi = (
        int(r[0] * scale_factor_x),
        int(r[1] * scale_factor_y),
        int(r[2] * scale_factor_x),
        int(r[3] * scale_factor_y)
    )

    # Crop the original image using the mapped coordinates
    cropped_image = img[original_roi[1]:original_roi[1] + original_roi[3],
                        original_roi[0]:original_roi[0] + original_roi[2]]

    # Remove the background from the cropped image
    cropped_bgrmv = remove(cropped_image)

    # Save the cropped image to a file
    output_path = "cropped_image.png"
    cv2.imwrite(output_path, cropped_bgrmv)

    print(f"Cropped image saved to: {output_path}")
    return cropped_bgrmv

def draw_outline(image_path):
    global drawing, ix, iy, img, img_copy

    drawing = False
    ix, iy = -1, -1
    img = cv2.imread(image_path)
    img_copy = img.copy()

    # Create a window for drawing an outline
    cv2.namedWindow('Draw Outline', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Draw Outline', draw_outline_callback)

    while True:
        cv2.imshow('Draw Outline', img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Clear the drawing and start over
            img = img_copy.copy()
        elif key == ord('q'):
            # Remove the background based on the drawn outline
            img = remove(img)
            cv2.imwrite('outlined_image.png', img)
            break

    cv2.destroyAllWindows()

def overlay_images(background_image_path, overlay_image_path):
    # Read the background and overlay images
    background = cv2.imread(background_image_path)
    overlay = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)

    # Convert images to grayscale
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)

    # Create ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors using ORB
    keypoints1, descriptors1 = orb.detectAndCompute(gray_background, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_overlay, None)

    # Create Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Get the best match
    best_match = matches[0]

    # Get coordinates of matched keypoints
    background_point = keypoints1[best_match.queryIdx].pt
    overlay_point = keypoints2[best_match.trainIdx].pt

    # Calculate position to overlay the image
    position = (int(background_point[0] - overlay_point[0]), int(background_point[1] - overlay_point[1]))

    # Overlay the images
    overlay_height, overlay_width, _ = overlay.shape
    roi = background[position[1]:position[1] + overlay_height, position[0]:position[0] + overlay_width]

    # Create a mask for the overlay image
    mask = overlay[:, :, 3] / 255.0
    mask_inv = 1.0 - mask
    mask_inv = mask_inv[:, :, np.newaxis]

    # Blend the images using NumPy indexing
    result_roi = (roi.astype(float) * mask_inv + overlay[:, :, :3].astype(float) * mask[:, :, np.newaxis]).astype(np.uint8)

    # Update the background image with the overlay
    background[position[1]:position[1] + overlay_height, position[0]:position[0] + overlay_width] = result_roi

    return background

def draw_outline_callback(event, x, y, flags, param):
    global drawing, ix, iy, img, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw a line during mouse movement
            cv2.line(img, (ix, iy), (x, y), (0, 255, 0), 5)
            ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Finalize the drawn line
        cv2.line(img, (ix, iy), (x, y), (0, 255, 0), 2)

if __name__ == "__main__":
    
    # change the file_name for performing the operation
    cropped_image = resize_and_select_roi("TEST_IMAGES/TEST IMAGES/2.jpg")
    
    draw_outline("cropped_image.png")

    # change the background_image_path filename for performing the operation
    background_image_path = "TEST_IMAGES/TEST IMAGES/2.jpg"
    overlay_image_path = "outlined_image.png"

    result_image = overlay_images(background_image_path, overlay_image_path)

    # Display the result
    cv2.namedWindow("Overlay Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Overlay Image", result_image)
    cv2.imwrite("overlay_2.png", result_image) # change the filename for the ouput image if needed

    # Remove temporary files
    os.remove("outlined_image.png")
    os.remove("cropped_image.png")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
