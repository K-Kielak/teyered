import cv2
import time

# Camera adjustment time
CAM_ADJUSTMENT_TIME = 0.1  # [s]

def record_video(debug):
    """
    Record the video using chosen mode
    :param debug: True displays the video being taken on the user's screen
    :return: List of frames (video)
    """
    video_frames = []
    cam = cv2.VideoCapture(0)

    # Start the video
    while(True):
        ret, frame = cam.read()
        if not ret:
            print("[teyered.image_collection.image_collection] Cam is not" + 
                  "setup correctly")
            time.sleep(0.1)
            continue

        video_frames.append(frame)

        if debug:
            cv2.imshow("Video", frame)
            # Stop the video display by pressing q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return video_frames
    

def take_photo(debug):
    """
    Take a photo using a chosen method
    :param debug: True displays the photo that is taken on the user's screen
    :return: Frame of the photo
    """
    cam = cv2.VideoCapture(0)
    time.sleep(CAM_ADJUSTMENT_TIME)
    ret, frame = cam.read()
    
    if not ret:
        print("[teyered.image_collection.image_collection] Cam is not" + 
              "setup correctly")

    if debug:
        cv2.imshow("Photo", frame)
        cv2.waitKey(0)

    cam.release()
    cv2.destroyAllWindows()
    return frame