import cv2
import filemanage

def get_video_fps_cv2(video_path):
    """
    Retrieves the FPS of a video file using OpenCV.

    Args:
        video_path (str): The path to the video file.

    Returns:
        float: The frames per second (FPS) of the video.
    """
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0.0

    # Get the FPS property
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Release the video capture object
    cap.release()
    
    return fps

# Example usage:
# filename = filemanage.select_video_file() # Replace with your video file path
# fps_value = get_video_fps_cv2(filename)
# print(f"Video FPS: {fps_value}")
