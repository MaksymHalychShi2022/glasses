import cv2
from PIL import Image
import torch
from inference import GlassesModelInference  # Importing from inference.py

if __name__ == "__main__":
    # Initialize OpenCV video capture
    cap = cv2.VideoCapture(0)  # Open the default webcam (0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Automatically determine the device if not specified
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GlassesModelInference(model_path="best_model.pth", device=device)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert the frame to a PIL image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run inference
        label = model.infer(pil_image)

        # Add the label text on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, label, (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the frame with the label
        cv2.imshow("Webcam Stream", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
