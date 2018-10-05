import cv2

def load_video(video_file_path, frames_to_skip, debug):
    """
    Load existing video file into the tool
    :param video_file_path: Full path to the video file
    :param frames_to_skip: Every nth frame will be analyzed
    :param debug: True prints some relevant video information to the console
    :return: Video frames
    """
    video_frames = []

    cap = cv2.VideoCapture(video_file_path)
    counter = -1  # Setting to -1 saves some hassle with lines

    # Video parameters
    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    video_fps = int(cap.get(5))
    video_frames_count = int(cap.get(7))

    if debug:
        print(f"\Loading video {video_file_path}:")
        print(f"Width (px): {str(video_width)}")
        print(f"Height (px): {str(video_height)}")
        print(f"FPS: {str(video_fps)}")
        print(f"Frames count: {str(video_frames_count)}")
        print(f"Analysing every {str(frames_to_skip)}th frame\n")

    while(cap.isOpened()):
        # Check if the video is over
        if cap.get(1) == video_frames_count:
            break

        ret, frame = cap.read()
        counter += 1

        if not ret:
            print("[teyered.data_collection.data_processing.process_video]" +
                  f"Frame {str(counter)} could not be read")
            continue

        # Showing only every n-th frame
        if (counter % frames_to_skip) != 0:
            continue

        frames.append(frame)

    if debug:
        print(f"Video loading for {video_file_path} has finished")

    cap.release()
    cv2.destroyAllWindows()
    return frames

def load_image(image_file_path, debug):
    """
    Load existing image from a specified file path into the tool
    :param image_file_path: Full file path to the image
    :param debug: Display the image to the user
    :return: Frame of the image
    """
    frame = cv2.imread(image_file_path)

    if debug:
        cv2.imshow(image_file_path, frame)
        cv2.waitKey(0)
    
    return frame