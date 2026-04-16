import cv2
from filemanage import select_video_file

def get_first_frame(video_path, output_image_path=r"img/first_frame5.jpg"):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    ret, frame = cap.read()

    if ret:
        cv2.imwrite(output_image_path, frame)
        print(f"First frame saved to {output_image_path}")
    else:
        print("Error: Could not read the first frame.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = select_video_file()
    get_first_frame(video_file)