import cv2
import numpy as np
import os
import shutil

# --- Configuration ---
OUTPUT_DIR = "sample_panorama_images"
MASTER_WIDTH = 4000
MASTER_HEIGHT = 1000
OUTPUT_IMAGE_WIDTH = 1200
OUTPUT_IMAGE_HEIGHT = 800
NUM_IMAGES_TO_GENERATE = 5 # <-- This is now the primary control

def create_master_landscape(width, height):
    """
    Programmatically creates a wide, master landscape image.
    (This function is unchanged from the original)
    """
    print("Generating master landscape image...")
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Sky Gradient
    for i in range(height // 2):
        color = (
            int(135 + (255 - 135) * (i / (height // 2))),
            int(206 + (255 - 206) * (i / (height // 2))),
            int(235 + (255 - 235) * (i / (height // 2)))
        )
        cv2.line(canvas, (0, i), (width, i), color, 1)

    # 2. Ground
    ground_color = (34, 139, 34)
    cv2.rectangle(canvas, (0, height // 2), (width, height), ground_color, -1)

    # 3. Mountains
    mountain_points = np.array([
        [(0, height // 2), (500, 200), (900, height // 2)],
        [(800, height // 2), (1500, 100), (2000, height // 2)],
        [(1800, height // 2), (2800, 350), (3500, height // 2)],
        [(3200, height // 2), (3700, 250), (width, height // 2)]
    ], np.int32)
    cv2.fillPoly(canvas, mountain_points, (105, 105, 105))

    # 4. Sun
    cv2.circle(canvas, (width - 600, 250), 80, (0, 191, 255), -1)
    cv2.circle(canvas, (width - 600, 250), 75, (255, 223, 0), -1)

    # 5. Trees
    tree_positions = [400, 1100, 1600, 2300, 3000, 3600]
    for x in tree_positions:
        cv2.rectangle(canvas, (x - 15, height - 400), (x + 15, height), (101, 67, 33), -1)
        cv2.circle(canvas, (x, height - 400), 60, (0, 100, 0), -1)
        cv2.circle(canvas, (x - 30, height - 380), 40, (0, 128, 0), -1)
        cv2.circle(canvas, (x + 30, height - 380), 40, (0, 128, 0), -1)

    # 6. Clouds
    cloud_positions = [(800, 150), (2000, 100), (3100, 180)]
    for (x, y) in cloud_positions:
        cv2.ellipse(canvas, (x, y), (80, 40), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(canvas, (x - 40, y), (60, 35), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(canvas, (x + 40, y), (60, 35), 0, 0, 360, (255, 255, 255), -1)

    print("Master landscape created.")
    return canvas

def extract_and_transform_images(master_img, num_images):
    """
    Extracts a specific number of overlapping photos from the master image.
    This version is robust and avoids infinite loops.
    """
    print(f"Extracting {num_images} images to '{OUTPUT_DIR}' directory...")
    
    # --- IMPROVEMENT 1: Clean the directory first ---
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    master_h, master_w = master_img.shape[:2]
    
    # --- IMPROVEMENT 2: Calculate step size based on desired image count ---
    # This ensures we get the exact number of images requested.
    step_size = (master_w - OUTPUT_IMAGE_WIDTH) // (num_images - 1)
    
    for i in range(num_images):
        # For the last image, ensure it captures the very edge of the master image
        if i == num_images - 1:
            start_x = master_w - OUTPUT_IMAGE_WIDTH
        else:
            start_x = i * step_size
        
        end_x = start_x + OUTPUT_IMAGE_WIDTH
        
        # 1. Extract the crop
        crop = master_img[:, start_x:end_x]
        
        # 2. Apply a slight random perspective warp
        h, w = crop.shape[:2]
        offset = 20
        
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = np.float32([
            [np.random.randint(-offset, offset), np.random.randint(-offset, offset)],
            [w + np.random.randint(-offset, offset), np.random.randint(-offset, offset)],
            [w + np.random.randint(-offset, offset), h + np.random.randint(-offset, offset)],
            [np.random.randint(-offset, offset), h + np.random.randint(-offset, offset)]
        ])
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_crop = cv2.warpPerspective(crop, M, (w, h))

        # 3. Save the final image with a 1-based index (01, 02, 03...)
        filename = os.path.join(OUTPUT_DIR, f"panorama_sample_{i+1:02d}.jpg")
        cv2.imwrite(filename, warped_crop)
        print(f"  -> Saved {filename}")

    print(f"\nSuccessfully generated {num_images} sample images.")


if __name__ == "__main__":
    master_image = create_master_landscape(MASTER_WIDTH, MASTER_HEIGHT)
    cv2.imwrite("master_landscape.jpg", master_image)
    print("\nMaster landscape saved as 'master_landscape.jpg' for reference.\n")

    # Pass the desired number of images to the function
    extract_and_transform_images(master_image, NUM_IMAGES_TO_GENERATE)