import cv2
from robomaster import robot
from robomaster import camera
from ultralytics import YOLOv10
from PIL import Image
import numpy as np

# Load YOLOv10 model
model = YOLOv10.from_pretrained('jameslahm/yolov10n')

def detect_person(frame):
    # Convert frame to PIL Image
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Perform detection
    results = model.predict(source=im, imgsz=640, conf=0.25, device='cuda')

    return results

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta")

    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis

    ep_camera.start_video_stream(display=True, resolution=camera.STREAM_360P)

    frame_width = 640
    frame_heght = 360

    while True:
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=5)
        results = detect_person(frame)

        # Extract person detections
        person_box = []
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # class 0 is 'person'
                    person_box.append(box.xywhn)  # Normalized center x, center y, width, height
        
        if len(person_box) != 0:
            w_max = 0
            x_w_max = 0.5
            for det_idx in range(len(person_box)):
                x, y, w, h = person_box[det_idx][0]
                
                # get x for the most wide box
                if w > w_max:
                    w_max = w
                    x_w_max = x

                center_x, center_y = int(x * frame_width), int(y * frame_heght)
                cv2.rectangle(frame, (int((x - w / 2) * frame_width), int((y - h / 2) * frame_heght)), 
                            (int((x + w / 2) * frame_width), int((y + h / 2) * frame_heght)), (255, 255, 255))
                cv2.putText(frame, "Person", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            cv2.imshow("Person Detection", frame)

            # Calculate the error from the center
            error_x = float(x) - 0.5
            if abs(error_x) > 0.05:
                # Rotate chassis to keep person in the center
                ep_chassis.drive_speed(x=0, y=0, z=error_x * 500)
            else:
                ep_chassis.drive_speed(x=0, y=0, z=0)
        else:
            ep_chassis.drive_speed(x=0, y=0, z=50)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    ep_camera.stop_video_stream()
    ep_robot.close()


