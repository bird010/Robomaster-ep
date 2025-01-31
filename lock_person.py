# script to rotate the robomaster ep to keep detected person in the middle of the camera frame.

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

def detect_person(frame):
    # Convert frame to PIL Image
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Perform detection
    results = model.predict(source=im, imgsz=640, conf=0.25, device='cuda')

    return results

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
                if(ep_status.get_front_dist() < 700):
                    # Rotate only
                    ep_chassis.drive_speed(x=0, y=0, z=error_x * 600)
                    ep_status.grip_close(ep_gripper)
                else:
                    # Rotate and chase, to keep person in the center
                    ep_chassis.drive_speed(x=0.5, y=0, z=error_x * 600)
                    ep_status.grip_open(ep_gripper)

            else:
                # person in the center
                if(ep_status.get_front_dist() < 700):
                    # stop
                    ep_chassis.drive_speed(x=0, y=0, z=0)
                    ep_status.grip_close(ep_gripper)
                else:
                    # Chase person
                    ep_chassis.drive_speed(x=0.7, y=0, z=0, timeout=5)
                    ep_status.grip_open(ep_gripper)
        else:
            # Searching person
            ep_chassis.drive_speed(x=0, y=0, z=50)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    ep_camera.stop_video_stream()
    ep_sensor.unsub_distance()
    ep_gripper.unsub_status()
    ep_robot.close()


