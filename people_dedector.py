import argparse
import cv2
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression

# Initialize OpenCV's HOG descriptor/person detector
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detector(image):
    """
    Detect people in an image using HOG + SVM and non-max suppression.

    Args:
        image (numpy.ndarray): The image in which to detect people.

    Returns:
        list: Bounding boxes of detected people after suppression.
    """
    clone = image.copy()
    (rects, weights) = HOGCV.detectMultiScale(image, winStride=(4, 4),
                                              padding=(8, 8), scale=1.05)
    # Apply non-max suppression
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    result = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    return result

def args_parser():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="People detection using OpenCV HOG + SVM")
    ap.add_argument("-i", "--image", default=None, help="Path to the image file")
    ap.add_argument("-c", "--camera", default=False, help="Use camera for detection (true/false)")
    ap.add_argument("-s", "--save", default=False, help="Save results to disk (true/false)")
    return vars(ap.parse_args())

def local_detect(image_path, save_result=False):
    """
    Detect people in a local image and optionally save the result.

    Args:
        image_path (str): Path to the image file.
        save_result (bool): Whether to save the result image to disk.

    Returns:
        tuple: (list of bounding boxes, result image)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image at {image_path}")
        return [], None

    image = imutils.resize(image, width=min(400, image.shape[1]))
    result = detector(image)
    
    counter = 0
    for (xA, yA, xB, yB) in result:
        counter += 1
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.putText(image, str(counter), (xA, yB), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_result:
        cv2.imwrite("result.jpg", image)
        print("[INFO] Result saved as result.jpg")
    
    return result, image

def camera_detect():
    """Detect people using the webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera")
            break

        frame = imutils.resize(frame, width=min(400, frame.shape[1]))
        result = detector(frame)

        counter = 0
        for (xA, yA, xB, yB) in result:
            counter += 1
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cv2.putText(frame, str(counter), (xA, yB), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Camera Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_people(args):
    """Main routine to detect people based on provided arguments."""
    camera_mode = str(args["camera"]).lower() == 'true'
    save_result = str(args["save"]).lower() == 'true'
    image_path = args["image"]

    if image_path and not camera_mode:
        print("[INFO] Reading local image...")
        local_detect(image_path, save_result=save_result)
    elif camera_mode:
        print("[INFO] Starting camera detection...")
        camera_detect()
    else:
        print("[ERROR] No image provided and camera mode not enabled.")

def main():
    args = args_parser()
    detect_people(args)

if __name__ == "__main__":
    main()
