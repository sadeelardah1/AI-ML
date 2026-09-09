import cv2

TARGET_SIZE = (224, 224)


def preprocess_image(path, target_size=TARGET_SIZE):
    """
    Shared Day 3 / Day 4 image preprocessing.

    Steps:
    1. Read image with OpenCV
    2. Resize to a fixed size
    3. Convert BGR to RGB
    4. Convert to float32
    5. Normalize pixels from 0-255 to 0-1
    """
    image_bgr = cv2.imread(str(path))

    if image_bgr is None:
        raise ValueError(f"Could not read image: {path}")

    image_bgr = cv2.resize(image_bgr, target_size)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype("float32") / 255.0

    return image_rgb
