import datetime
import cv2
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import sys

# Create the dataset
def create_dataset(image, SHOW_IMAGE):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../frame_extractor/shape_predictor_68_face_landmarks.dat')

    # Lists that contain the points for the right and left eyes
    right_eye = []
    left_eye = []

    # Transform the picture into a gray picture
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Copy the original image to draw on
        if SHOW_IMAGE:
            clone = image.copy()

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

            # Add the points for right and left eyes
            for (x, y) in shape[i:j]:
                if (i, j) == (36,42): # Right eye
                    right_eye.append((x,y))
                    if SHOW_IMAGE:
                        cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                elif (i, j) == (42,48): # Left eye
                    left_eye.append((x,y))
                    if SHOW_IMAGE:
                        cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

        # Show images if set to true
        if SHOW_IMAGE:
            cv2.imshow('image.png', clone)
            cv2.waitKey(0)

    # Write eye data to csv file
    image_name_no_extension = IMAGE_NAME.split(".")[0]
    f = open(f'face_points_{image_name_no_extension}.csv','w')
    for i in range(0, len(right_eye)):
        if i == 0:
            f.write("x,y\n")
        f.write(f"{str(right_eye[i][0])},{str(right_eye[i][1])}\n")
    for i in range(0, len(left_eye)):
        f.write(f"{str(left_eye[i][0])},{str(left_eye[i][1])}\n")
    f.close()

# Run the application
def run_extractor(IMAGE_NAME):
    print(f"Currently working with file: {IMAGE_NAME}")

    # Parameters
    SHOW_IMAGE = True

    # Open the image
    image = cv2.imread(IMAGE_NAME)

    # Process
    create_dataset(image, SHOW_IMAGE)

    # End
    print(f"Work with the file {IMAGE_NAME} is finished")

# python3 main.py image.jpeg
if __name__ == '__main__':
    IMAGE_NAME = str(sys.argv[1])
    run_extractor(IMAGE_NAME)
