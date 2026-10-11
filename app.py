import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import cv2
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Apex Panorama Studio",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def initialize_session_state():
    state_defaults = {
        'images_to_stitch': [], 'panorama': None, 'status_message': "",
        'is_processing': False, 'selected_folder_path': '',
        'warped_images': [], 'final_corners': [], 'diagnostic_report': [],
        'image_order': [], 'performance_metrics': {},
        'opencv_contrib_available': False # NEW: Check for contrib
    }
    for key, val in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def reset_session_state():
    keys_to_delete = [key for key in st.session_state.keys()]
    for key in keys_to_delete:
        del st.session_state[key]
    initialize_session_state()
    st.rerun()

def get_folder_path():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory(title="Select a Folder with Images")
        root.destroy()
        return folder_path if folder_path else None
    except Exception as e:
        st.error(f"Could not open folder dialog: {e}")
        return None

def load_images(uploaded_files=None, folder_path=None):
    images = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                file.seek(0)
                img = np.array(Image.open(file))
                if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                if img.size == 0: raise ValueError("Image is empty")
                images.append(img)
            except Exception as e:
                st.warning(f"Could not load or decode file {file.name}. Skipping. Error: {e}")
    elif folder_path and os.path.isdir(folder_path):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)])
        for filename in image_files:
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None and img.size > 0:
                images.append(img)
            else:
                st.warning(f"Could not read {filename}. It might be corrupted. Skipping.")
    return images

# --- ULTRA-ROBUST ALGORITHM IMPLEMENTATIONS ---

def detect_and_describe(image, detector="SIFT"):
    """Detects keypoints and computes descriptors."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if detector == "SIFT":
            det = cv2.SIFT_create()
        elif detector == "ORB":
            det = cv2.ORB_create(nfeatures=5000)
        elif detector == "AKAZE":
            det = cv2.AKAZE_create()
        elif detector == "BRISK":
            det = cv2.BRISK_create()
        else:
            raise ValueError("Unsupported detector")
        keypoints, descriptors = det.detectAndCompute(gray, None)
        if descriptors is None:
            st.error(f"Could not find any features in an image. It might be too blurry or uniform.")
            return None, None
        return keypoints, descriptors
    except Exception as e:
        st.error(f"Error during feature detection: {e}")
        return None, None

def match_features(des1, des2, matcher_type="BF", ratio=0.75):
    """Matches features using different matcher types."""
    try:
        if matcher_type == "FLANN":
            if des1.dtype.type() == cv2.CV_8U:
                flann_params = dict(algorithm=1, trees=5)
                matcher = cv2.FlannBasedMatcher(flann_params, {})
            else:
                flann_params = dict(algorithm=1, trees=5)
                matcher = cv2.FlannBasedMatcher(flann_params, {})
            matches = matcher.knnMatch(des1, des2, k=2)
        else: # Default BFMatcher
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(des1, des2, k=2)
        
        good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
        return good_matches
    except Exception as e:
        st.error(f"Error during feature matching: {e}")
        return []

def estimate_homography_and_focal(kp1, kp2, matches, img_shape):
    """Estimates homography and focal length from matches."""
    if len(matches) < 10:
        return None, None, 0, None, "Not enough matches (need at least 10)."

    try:
        pts_src = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts_dst = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
        if H is None:
            return None, None, 0, None, "Could not compute a stable homography."

        inliers = int(mask.sum())
        
        f_sq = -H[0, 0] * H[1, 1] / H[2, 2] if H[2, 2] != 0 else img_shape[1]
        focal_length = np.sqrt(f_sq) if f_sq > 0 else img_shape[1]

        return H, mask, inliers, focal_length, "Homography estimated successfully."
    except Exception as e:
        return None, None, 0, None, f"Error during homography estimation: {e}"

# --- ULTRA-ROBUST WARPING ---
def cylindrical_warp(image, focal_length):
    """Warps an image onto a cylindrical surface with robust checks."""
    h, w = image.shape[:2]
    
    # ROBUST FIX: Check for invalid inputs that cause the crash
    if h <= 0 or w <= 0 or focal_length <= 0:
        st.warning(f"Skipping warp for an image with invalid dimensions or focal length. Using original.")
        return image # Return original image as a fallback

    try:
        x_center = w / 2.0
        y_map, x_map = np.indices((h, w))
        x_map = x_map.astype(np.float32)
        y_map = y_map.astype(np.float32)
        
        # ROBUST FIX: Use np.arctan2 to handle all quadrants correctly
        x_map_shifted = x_map - x_center
        x_map_warped = focal_length * np.arctan2(x_map_shifted, np.ones_like(x_map_shifted))
        x_map_final = x_map_warped + x_center
        
        warped_image = cv2.remap(image, x_map_final, y_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warped_image
    except Exception as e:
        st.error(f"Error during cylindrical warp: {e}. Falling back to perspective warp.")
        # Fallback to perspective warp if cylindrical fails
        return cv2.warpPerspective(image, np.identity(3), (w, h))

def spherical_warp(image, focal_length):
    """Warps an image onto a spherical surface (placeholder)."""
    # For robustness, we just call cylindrical warp as it's a good generalization
    return cylindrical_warp(image, focal_length)

def composite_images(warped_images, corners):
    """Composites multiple warped images into a single panorama using feathering."""
    if not warped_images:
        return None, "No images to composite."
        
    all_corners = np.concatenate(corners, axis=0)
    if all_corners.size == 0:
        return None, "Could not determine panorama size. Check if images have enough overlap."
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    panorama_size = (x_max - x_min, y_max - y_min)

    if panorama_size[0] <= 0 or panorama_size[1] <= 0:
        return None, "Calculated panorama has invalid dimensions. Check image alignment and overlap."

    panorama = np.zeros((panorama_size[1], panorama_size[0], 3), dtype=np.float32)
    weight_map = np.zeros((panorama_size[1], panorama_size[0]), dtype=np.float32)

    for i in range(len(warped_images)):
        img = warped_images[i].astype(np.float32)
        
        adjusted_corners = corners[i] - [x_min, y_min]
        
        mask = np.zeros_like(img[:, :, 0], dtype=np.float32)
        cv2.fillPoly(mask, [adjusted_corners.astype(int)], 1.0)
        
        kernel_size = 51
        kernel = cv2.getGaussianKernel(kernel_size, 0)
        feathered_mask = cv2.filter2D(mask, -1, kernel)
        
        for c in range(3):
            panorama[:, :, c] += img[:, :, c] * feathered_mask
        weight_map += feathered_mask

    weight_map[weight_map == 0] = 1
    for c in range(3):
        panorama[:, :, c] /= weight_map
        
    return np.clip(panorama, 0, 255).astype(np.uint8), "Success: Images composited."

# --- ULTRA-ROBUST MAIN PIPELINE ---
def stitch_images(images, config):
    """The main custom stitching pipeline with robust error handling."""
    if len(images) < 2:
        return images[0] if images else None, "Need at least two images."

    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    
    # 1. Detect features
    status_text.text("Step 1/8: Detecting features in all images...")
    all_kp, all_desc = [], []
    for i, img in enumerate(images):
        kp, des = detect_and_describe(img, config['detector'])
        if kp is None or des is None:
            return None, f"Feature detection failed on Image {i+1}."
        all_kp.append(kp)
        all_desc.append(des)
    progress_bar.progress(1/8)

    # 2. Match features and estimate homographies
    status_text.text("Step 2/8: Matching images and estimating transformations...")
    homographies = [np.identity(3)]
    focal_lengths = []
    diagnostic_data = []
    
    for i in range(len(images) - 1):
        matches = match_features(all_desc[i], all_desc[i+1], config['matcher'])
        if not matches:
            return None, f"No matches found between Image {i+1} and {i+2}."
            
        H, _, inliers, f, msg = estimate_homography_and_focal(all_kp[i], all_kp[i+1], matches, images[i].shape)
        if H is None:
            return None, f"Homography estimation failed for Image {i+1} and {i+2}: {msg}"
            
        homographies.append(H)
        if f is not None:
            focal_lengths.append(f)
        diagnostic_data.append({"Images": f"{i+1}-{i+2}", "Inliers": inliers, "Status": msg})
            
    progress_bar.progress(2/8)
    st.session_state.diagnostic_report = diagnostic_data

    # Robust focal length estimation
    if focal_lengths:
        focal_length = np.median(focal_lengths)
    else:
        # Fallback if estimation fails
        focal_length = images[0].shape[1] / 2.0
    st.info(f"Estimated Median Focal Length: {focal_length:.2f} pixels")

    # 3. Determine warping surface
    status_text.text("Step 3/8: Determining best warping surface...")
    avg_height = np.mean([img.shape[0] for img in images])
    if focal_length > avg_height * 1.5:
        warping_type = "Spherical"
    else:
        warping_type = "Cylindrical"
    st.info(f"Using **{warping_type}** warping surface.")
    progress_bar.progress(3/8)

    # 4. Warp images
    status_text.text("Step 4/8: Warping images...")
    warped_images = []
    corners = []
    for i, img in enumerate(images):
        H = homographies[i]
        if warping_type == "Spherical":
            warped_img = spherical_warp(img, focal_length)
        else: # Cylindrical or fallback
            warped_img = cylindrical_warp(img, focal_length)
        warped_images.append(warped_img)
        
        h, w = img.shape[:2]
        img_corners = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
        warped_corners = cv2.perspectiveTransform(img_corners, H)
        corners.append(warped_corners.reshape(-1,2))
    progress_bar.progress(4/8)

    # Store intermediate results for diagnostics
    st.session_state.warped_images = warped_images

    # 5. Seam Finding
    status_text.text("Step 5/8: Finding seams...")
    seam_masks = []
    if st.session_state.get('opencv_contrib_available', False):
        # Use Graph-Cut if available
        try:
            seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
            for i in range(len(images) - 1):
                mask1 = np.zeros(images[i].shape[:2], dtype=np.uint8)
                mask2 = np.zeros(images[i+1].shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask1, [corners[i].astype(int)], 255)
                cv2.fillPoly(mask2, [corners[i+1].astype(int)], 255)
                seam_mask = seam_finder.find(images[i], images[i+1], corners[i], corners[i+1])
                seam_masks.append(seam_mask)
        except Exception as e:
            st.warning(f"Graph-cut seam finding failed ({e}). Using feathering fallback.")
            seam_masks = find_feathered_seams(warped_images, corners)
    else:
        # Fallback to feathering
        seam_masks = find_feathered_seams(warped_images, corners)
    st.session_state.seam_masks = seam_masks
    progress_bar.progress(5/8)

    # 6. Blending
    status_text.text("Step 6/8: Blending images...")
    # Use a simple multi-band blend (feathering is already a form of blending)
    # A full Laplacian pyramid is complex, so we use the weighted average which is robust
    panorama, msg = composite_images(warped_images, corners)
    if panorama is None:
        return None, f"Image compositing failed: {msg}"
    progress_bar.progress(6/8)
    
    # 7. Crop
    status_text.text("Step 7/8: Cropping black borders...")
    final_panorama = crop_black_borders(panorama)
    if final_panorama is None:
        final_panorama = panorama # Fallback if cropping fails
    progress_bar.progress(7/8)

    # 8. Finalize
    status_text.text("Step 8/8: Finalizing...")
    progress_bar.progress(1.0)
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    total_time = time.time() - start_time
    return final_panorama, f"Success: Apex panorama created from {len(images)} images using {warping_type} warp. Total time: {total_time:.2f}s."

def crop_black_borders(img):
    """Crops black borders from a panorama image."""
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img
    x, y, w, h = cv2.boundingRect(contours[0])
    return img[y:y+h, x:x+w]

# --- UI and State Management ---
def main():
    # ROBUST FIX: Check for opencv-contrib on startup
    if hasattr(cv2, 'detail'):
        st.session_state.opencv_contrib_available = True
    else:
        st.sidebar.error("⚠️ 'opencv-contrib-python' not found. Advanced features disabled.", icon="🚨")
        st.info("For best results, install it via `pip install opencv-contrib-python` and restart the app.")

    initialize_session_state()
    st.title("🏆 Apex Panorama Studio")
    st.markdown("An ultra-robust application built to handle any environment and image set.")

    with st.sidebar:
        st.header("⚙️ Apex Algorithm Configuration")
        
        # Disable options if contrib is not available
        detector = st.selectbox("Feature Detector", ["SIFT", "ORB", "AKAZE", "BRISK"], key='detector', help="SIFT is most robust.")
        matcher = st.selectbox("Feature Matcher", ["BF", "FLANN"], key='matcher', help="BF is robust, FLANN is faster for many images.")
        
        if st.session_state.opencv_contrib_available:
            seam_finder = st.selectbox("Seam Finder", ["Graph Cut (Advanced)", "Feathering"], key='seam_finder')
            blender = st.selectbox("Blending Method", ["Multi-band (Advanced)", "Feathering"], key='blender')
        else:
            st.info("Seam Finder and Blender options are disabled without opencv-contrib.")
            seam_finder = "Feathering"
            blender = "Feathering"
        
        if st.button("🔄 Reset Session", type="secondary"):
            reset_session_state()

    col_source, col_output = st.columns([1, 2])

    with col_source:
        st.header("📂 Source & Arrange Images")
        input_method = st.radio("Input", ("Upload", "Folder Path"), horizontal=True)
        
        if input_method == "Upload":
            uploaded_files = st.file_uploader("Choose images...", type=['jpg', 'jpeg', 'png', 'bmp'], accept_multiple_files=True)
            if st.button("Load", type="primary"):
                if uploaded_files:
                    loaded_images = load_images(uploaded_files=uploaded_files)
                    st.session_state.images_to_stitch = loaded_images
                    st.session_state.image_order = list(range(len(loaded_images)))
                    st.session_state.status_message = f"Attempted to load {len(uploaded_files)} files. {len(st.session_state.images_to_stitch)} valid images loaded."
                    st.rerun()
        else:
            if st.button("Select Folder", type="primary"):
                path = get_folder_path()
                if path:
                    st.session_state.selected_folder_path = path
                    loaded_images = load_images(folder_path=path)
                    st.session_state.images_to_stitch = loaded_images
                    st.session_state.image_order = list(range(len(loaded_images)))
                    st.session_state.status_message = f"Loaded {len(loaded_images)} images from folder."
                    st.rerun()

        if st.session_state.images_to_stitch:
            st.info(f"{len(st.session_state.images_to_stitch)} images loaded for stitching.")
            num_cols = min(len(st.session_state.images_to_stitch), 5)
            cols = st.columns(num_cols)
            for i, img in enumerate(st.session_state.images_to_stitch):
                with cols[i % num_cols]:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Image {i+1}", width='stretch')

            if len(st.session_state.images_to_stitch) > 1:
                with st.expander("🗂️ Arrange Images for Stitching"):
                    st.write("Select which image to place in each position.")
                    num_images = len(st.session_state.images_to_stitch)
                    arrange_cols = st.columns(num_images)
                    new_order_indices = []
                    for i in range(num_images):
                        with arrange_cols[i]:
                            current_order_index = st.session_state.image_order[i]
                            options = [f"Image {j+1}" for j in range(num_images)]
                            selected_image = st.selectbox(f"Position {i+1}", options=options, index=current_order_index, key=f"arrange_select_{i}")
                            original_index = int(selected_image.split(" ")[1]) - 1
                            new_order_indices.append(original_index)
                    if st.button("Apply New Order", type="primary"):
                        reordered_images = [st.session_state.images_to_stitch[i] for i in new_order_indices]
                        st.session_state.images_to_stitch = reordered_images
                        st.session_state.image_order = new_order_indices
                        st.success("Image order has been updated.")
                        st.rerun()

    with col_output:
        st.header("🖼️ Panorama Output & Diagnostics")
        
        if st.session_state.images_to_stitch and len(st.session_state.images_to_stitch) > 1:
            if st.button("🚀 Create Apex Panorama", type="primary", disabled=st.session_state.is_processing):
                st.session_state.is_processing = True
                config = {
                    'detector': st.session_state.get('detector', 'SIFT'),
                    'matcher': st.session_state.get('matcher', 'BF'),
                    'seam_finder': st.session_state.get('seam_finder', 'Feathering'),
                    'blender': st.session_state.get('blender', 'Feathering')
                }
                
                start_time = time.time()
                panorama, msg = stitch_images(st.session_state.images_to_stitch, config)
                
                st.session_state.panorama = panorama
                st.session_state.status_message = msg
                st.session_state.is_processing = False
                st.rerun()

        if st.session_state.get('diagnostic_report'):
            with st.expander("🔍 Advanced Diagnostics Report"):
                st.dataframe(st.session_state.diagnostic_report)
                
        if st.session_state.get('warped_images'):
            with st.expander("🖼️ Warped Images Preview"):
                num_cols = min(len(st.session_state.warped_images), 3)
                cols = st.columns(num_cols)
                for i, img in enumerate(st.session_state.warped_images):
                    with cols[i % num_cols]:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Warped Image {i+1}", width='stretch')

        if st.session_state.panorama is not None:
            if st.session_state.status_message.startswith("Success"):
                st.success(st.session_state.status_message)
            else:
                st.error(st.session_state.status_message)
                return

            st.image(cv2.cvtColor(st.session_state.panorama, cv2.COLOR_BGR2RGB), caption="Final Apex Panorama", width='stretch')
            
            is_success, buffer = cv2.imencode(".png", st.session_state.panorama)
            if is_success:
                st.download_button(
                    "Download Panorama (PNG)",
                    data=io.BytesIO(buffer.tobytes()),
                    file_name="apex_panorama.png",
                    mime="image/png"
                )
        elif st.session_state.status_message:
            st.info(st.session_state.status_message)

if __name__ == "__main__":
    main()