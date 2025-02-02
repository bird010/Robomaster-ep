# script to rotate the robomaster ep to keep detected Cup in the middle of the camera frame.

import cv2
from robomaster import robot
from robomaster import camera
from ultralytics import YOLOv10
from PIL import Image
import numpy as np

# Load YOLOv10 model
model = YOLOv10.from_pretrained('jameslahm/yolov10n')

class EpStatus:
    def __init__(self):
        self._front_dist = 0
        self._grip_status = 'normal'

    def set_front_dist(self, distance):
        self._front_dist = distance

    def get_front_dist(self):
        return self._front_dist

    def dist_sub_data_handler(self, sub_info):
        distance = sub_info
        self.set_front_dist(distance[0])
        print("tof1:{0}".format(distance[0]))

    def set_grip_status(self, status):
        self._grip_status = status

    def get_grip_status(self):
        return self._grip_status
    
    def grip_open(self, ep_gripper):
        if self.get_grip_status() != 'opened':
            ep_gripper.open()

    def grip_close(self, ep_gripper):
        if self.get_grip_status() != 'closed':
            ep_gripper.close()

    def grip_sub_data_handler(self, sub_info):
        # 完全闭合 closed, 完全张开opened, 处在中间位置normal.
        status = sub_info
        self.set_grip_status(status)
        print("gripper status:{0}.".format(status))

def detect(frame, obj_str):
    # Convert frame to PIL Image
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Perform detection
    results = model.predict(source=im, imgsz=640, conf=0.25, device='cuda')

    det_dict = {'person': 0, 
                'bicycle': 1,
                'car': 2,
                'motorcycle': 3,
                'airplane': 4,
                'bus': 5,
                'train': 6,
                'truck': 7,
                'boat': 8,
                'traffic light': 9,
                'fire hydrant': 10,
                'stop sign': 11,
                'parking meter': 12,
                'bench': 13,
                'bird': 14,
                'cat': 15,
                'dog': 16,
                'horse': 17,
                'sheep': 18,
                'cow': 19,
                'elephant': 20,
                'bear': 21,
                'zebra': 22,
                'giraffe': 23,
                'backpack': 24,
                'umbrella': 25,
                'handbag': 26,
                'tie': 27,
                'suitcase': 28,
                'frisbee': 29,
                'skis': 30,
                'snowboard': 31,
                'sports ball': 32,
                'kite': 33,
                'baseball bat': 34,
                'baseball glove': 35,
                'skateboard': 36,
                'surfboard': 37,
                'tennis racket': 38,
                'bottle': 39,
                'wine glass': 40,
                'cup': 41,
                'fork': 42,
                'knife': 43,
                'spoon': 44,
                'bowl': 45,
                'banana': 46,
                'apple': 47,
                'sandwich': 48,
                'orange': 49,
                'broccoli': 50,
                'carrot': 51,
                'hot dog': 52,
                'pizza': 53,
                'donut': 54,
                'cake': 55,
                'chair': 56,
                'couch': 57,
                'potted plant': 58,
                'bed': 59,
                'dining table': 60,
                'toilet': 61,
                'tv': 62,
                'laptop': 63,
                'mouse': 64,
                'remote': 65,
                'keyboard': 66,
                'cell phone': 67,
                'microwave': 68,
                'oven': 69,
                'toaster': 70,
                'sink': 71,
                'refrigerator': 72,
                'book': 73,
                'clock': 74,
                'vase': 75,
                'scissors': 76,
                'teddy bear': 77,
                'hair drier': 78,
                'toothbrush': 79,
            }

    # Extract detections
    result_box = []
    for result in results:
        for box in result.boxes:
            if box.cls == det_dict[obj_str]:  # class 0 is 'person'
                result_box.append(box.xywhn)  # Normalized center x, center y, width, height

    return result_box

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta")

    ep_status = EpStatus()

    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis
    ep_gripper = ep_robot.gripper
    ep_gripper.sub_status(freq=5, callback=ep_status.grip_sub_data_handler)
    ep_sensor = ep_robot.sensor
    ep_sensor.sub_distance(freq=20, callback=ep_status.dist_sub_data_handler)
    
    ep_camera.start_video_stream(display=True, resolution=camera.STREAM_360P)

    frame_width = 640
    frame_heght = 360

    while True:
        det_str = 'bottle'
        frame = ep_camera.read_cv2_image(strategy="newest", timeout=5)
        box = detect(frame, det_str)
        
        if len(box) != 0:
            w_max = 0
            x_w_max = 0.5
            for det_idx in range(len(box)):
                x, y, w, h = box[det_idx][0]
                
                # get x for the most wide box
                if w > w_max:
                    w_max = w
                    x_w_max = x

                center_x, center_y = int(x * frame_width), int(y * frame_heght)
                cv2.rectangle(frame, (int((x - w / 2) * frame_width), int((y - h / 2) * frame_heght)), 
                            (int((x + w / 2) * frame_width), int((y + h / 2) * frame_heght)), (255, 255, 255))
                cv2.putText(frame, det_str, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            cv2.imshow(det_str, frame)

            # Calculate the error from the center
            error_x = float(x_w_max) - 0.5
            if abs(error_x) > 0.05:
                # Cup not in the center
                if(ep_status.get_front_dist() < 700):
                    # Rotate only
                    ep_chassis.drive_speed(x=0, y=0, z=error_x * 200)
                else:
                    # Rotate and chase, to keep Cup in the center
                    ep_chassis.drive_speed(x=0, y=0, z=error_x * 200)
                    ep_status.grip_open(ep_gripper)

            else:
                # Cup in the center
                if(ep_status.get_front_dist() < 700):
                    # stop
                    ep_chassis.drive_speed(x=0, y=0, z=0)
                    ep_status.grip_close(ep_gripper)
                else:
                    # Chase Cup
                    ep_chassis.drive_speed(x=0.3, y=0, z=0, timeout=5)
                    ep_status.grip_open(ep_gripper)
        else:
            # Searching Cup
            ep_chassis.drive_speed(x=0, y=0, z=50)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    ep_camera.stop_video_stream()
    ep_sensor.unsub_distance()
    ep_gripper.unsub_status()
    ep_robot.close()


