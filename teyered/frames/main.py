import cv2
from imutils import face_utils
import numpy as np
import imutils
import dlib
import sys

# Calculate the area of the polygon
def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

# Calculate max y values
def max_y_values(points):
    y_values = []
    for point in points:
        y_values.append(point[1])
    y_values.sort()
    distance = y_values[len(y_values)-1] - y_values[0]
    return distance

# Create the dataset
def create_area_data_set(images, SHOW_IMAGES, PRINT_INFO):

    print("Creating the area dataset...\n")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    area_data = []
    y_data = []

    RESIZE_WIDTH = 250

    for image in images:
        # Resize the image
        image = imutils.resize(image, width=500)

        # Transform the picture into a gray picture
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):

            # Lists for points that describe area
            right_area = []
            left_area = []

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                # Keep a clone of the original image
                clone = image.copy()

                # Add the points for right and left eyes
                for (x, y) in shape[i:j]:
                    if (i, j) == (36,42): # Right eye
                        right_area.append((x,y))
                    elif (i, j) == (42,48): # Left eye
                        left_area.append((x,y))

                # Masks
                if (i,j) == (36,42):
                    # Masking to make it black
                    mask = np.zeros(image.shape, dtype=np.uint8)
                    roi_corners = np.array([right_area], dtype=np.int32)
                    channel_count = image.shape[2]
                    ignore_mask_color = (255,)*channel_count
                    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
                    masked_image = cv2.bitwise_and(image, mask)

                    # Actual roi and the points
                    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                    roi = masked_image[y:y + w, x:x + w]

                    # Normalising to get from 0 to actual range
                    normalised_right_area = []
                    for i in range(0, len(right_area)):
                        normalised_right_area_x = right_area[i][0] - x
                        normalised_right_area_y = right_area[i][1] - y
                        normalised_right_area.append((normalised_right_area_x, normalised_right_area_y))

                    # Resizing the points to fit the universal measuring
                    resized_right_area = []
                    resized_x_len = w
                    resized_y_len = w
                    for i in range(0, len(normalised_right_area)):
                        resized_right_area_x = int((normalised_right_area[i][0] * RESIZE_WIDTH) / resized_x_len)
                        resized_right_area_y = int((normalised_right_area[i][1] * RESIZE_WIDTH) / resized_y_len)
                        resized_right_area.append((resized_right_area_x, resized_right_area_y))

                    # Resize the photo
                    roi = imutils.resize(roi, width=RESIZE_WIDTH, inter=cv2.INTER_CUBIC)

                    # Show images if set to true
                    if SHOW_IMAGES:
                        cv2.imshow('image_masked_right.png', roi)

                # Literally the same
                elif (i,j) == (42,48):
                    mask = np.zeros(image.shape, dtype=np.uint8)
                    roi_corners = np.array([left_area], dtype=np.int32)
                    channel_count = image.shape[2]
                    ignore_mask_color = (255,)*channel_count
                    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
                    masked_image = cv2.bitwise_and(image, mask)

                    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                    roi = masked_image[y:y + w, x:x + w]

                    normalised_left_area = []
                    for i in range(0, len(left_area)):
                        normalised_left_area_x = left_area[i][0] - x
                        normalised_left_area_y = left_area[i][1] - y
                        normalised_left_area.append((normalised_left_area_x, normalised_left_area_y))

                    resized_left_area = []
                    resized_x_len = w
                    resized_y_len = w
                    for i in range(0, len(normalised_left_area)):
                        resized_left_area_x = int((normalised_left_area[i][0] * RESIZE_WIDTH) / resized_x_len)
                        resized_left_area_y = int((normalised_left_area[i][1] * RESIZE_WIDTH) / resized_y_len)
                        resized_left_area.append((resized_left_area_x, resized_left_area_y))

                    roi_left = imutils.resize(roi, width=RESIZE_WIDTH, inter=cv2.INTER_CUBIC)

                    if SHOW_IMAGES:
                        cv2.imshow('image_masked_left.png', roi_left)
                if SHOW_IMAGES:
                    cv2.imshow("Clone1.png", clone)

        # All the calculations of the area
        right_area_f = PolygonArea(resized_right_area)
        left_area_f = PolygonArea(resized_left_area)

        # Calculations of y values
        right_y_value = max_y_values(resized_right_area)
        left_y_value = max_y_values(resized_left_area)

        area_data.append([right_area_f, left_area_f])
        y_data.append([right_y_value, left_y_value])

        # Some debugging stuff
        if PRINT_INFO:
            print("Resized right area: " + str(right_area_f))
            print("Resized left area: " + str(left_area_f))
            print("Resized right y value: " + str(right_y_value))
            print("Resized left y value: " + str(left_y_value))
            print("Right points: " + str(resized_right_area))
            print("Left points: " + str(resized_left_area))
            print("") # Empty line

        # Skip to the next image after pressing
        cv2.waitKey(0)

    # Actual np arrays
    area_data = np.array(area_data)
    y_data = np.array(y_data)

    # Write training data to csv file
    print("Writing to the files...\n")

    f = open('area_data.csv','w')
    for i in range(0, area_data.shape[0]):
        if i == 0:
            f.write("right_area,left_area\n")
        f.write(str(area_data[i][0]) + "," + str(area_data[i][1]) + "\n")
    f.close()

    f = open('y_data.csv','w')
    for i in range(0, y_data.shape[0]):
        if i == 0:
            f.write("right,left\n")
        f.write(str(y_data[i][0]) + "," + str(y_data[i][1]) + "\n")
    f.close()

# Interpreting the video, converting it into the frames and preparing for the analysis
def video_intepreter(VIDEO_TITLE, FRAMES_TO_SKIP):
    cap = cv2.VideoCapture(VIDEO_TITLE)
    counter = -1 # -1 instead of 0 saves one line of code lmao
    frames = [] # List holding all of the frames from the video

    # Video parameters
    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    video_fps = int(cap.get(5))
    video_frames_count = int(cap.get(7))

    print("\n")
    print("Width of the video (px): " + str(video_width))
    print("Height of the video (px): " + str(video_height))
    print("FPS of the video: " + str(video_fps))
    print("Frames count of the video: " + str(video_frames_count))
    print("Analysing every " + str(FRAMES_TO_SKIP) + "th frame")
    print("\n")

    while(cap.isOpened()):
        # Check if the video is over
        if cap.get(1) == video_frames_count:
            break

        ret, frame = cap.read() # ret - return value, frame - numpy array of the image
        counter += 1

        # Check if everything is ok
        if not ret:
            print("Error: frame " + str(counter) + " could not be read")
            continue

        # Showing only every n-th frame
        if (counter % FRAMES_TO_SKIP) != 0:
            continue

        frames.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    print("Video interpreting finished\n")
    return frames

# Run the application
def run_tired(VIDEO_TITLE):
    print("Currently working with video file: " + VIDEO_TITLE)

    # Parameters
    FRAMES_TO_SKIP = 2  # Every n-th frame will be analysed
    SHOW_IMAGES = False
    PRINT_INFO = False

    # Process
    frames = video_intepreter(VIDEO_TITLE, FRAMES_TO_SKIP)
    create_area_data_set(frames, SHOW_IMAGES, PRINT_INFO)

    print("Work with the video file is finished")

# python3 main.py clip.mp4
if __name__ == '__main__':
    VIDEO_TITLE = str(sys.argv[1])
    run_tired(VIDEO_TITLE)
