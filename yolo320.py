# import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
from pymycobot import MyCobot320,utils
import threading
import time
import transforms3d as tf3d
class Camera:
    def __init__(self,index=1):
        self.cap = cv2.VideoCapture(index)
        

        self.cameraMatrix =np.array([[532.76634274  , 0.    ,     306.22017641],
        [  0.      ,   532.94411762 ,225.40097394],
        [  0.    ,       0.    ,       1.    ,    ]])


        self.distCoeffs = np.array([[ 0.17156132, -0.66222681, -0.00294354 , 0.00106322 , 0.73823942]])
        if not self.cap.isOpened():
            print("no")
            exit()
    
    def get_frame(self):
        while True:
           
           
            ret, frame =self.cap.read()

            
            if not ret:
                print("no")
                break

           
            return frame
        
    def get_camera_coordinate(self,u,v,Z=0.234):
        points = np.array([u, v], dtype=np.float32)
        undistorted = cv2.undistortPoints(points, self.cameraMatrix,self.distCoeffs)
        # print("undistorted=",undistorted)
        x_norm = undistorted[0,0,0]
        y_norm = undistorted[0,0,1]
        Zc=(Z-30)/1000
        # print("ZC=",Zc)
        Xc = float(round(x_norm * Zc *1000,2))
        Yc = float(round(y_norm * Zc *1000,2))
        camera_coords = [Xc,Yc]
        return camera_coords
       

            

        

          
class Yolo:
    lock = threading.Lock()
    def __init__(self,model_path,class_names_path) :
        self.show_list=[]
        self.CONFIDENCE_THRESHOLD = 0.4

        self.SCORE_THRESHOLD = 0.5
        self.NMS_THRESHOLD = 0.45

        self.BLACK = (0, 0, 0)
        self.BLUE = (255, 178, 50)
        self.YELLOW = (0, 255, 255)

        self.FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.7
        self.THICKNESS = 1
        self.HEIGHT = 640
        self.WIDTH = 640

        # dir_prefix = Path("realsense-kit-main\resources\yolo")
        # modelWeights = str(dir_prefix / "yolov5s.onnx")
        modelWeights =model_path
        self.net = cv2.dnn.readNet(modelWeights)

        # classesFile = dir_prefix / "coco.names"
        classesFile = class_names_path
       
        with open(classesFile, "rt") as f:
            self.classes = f.read().rstrip("\n").split("\n")


    def detect(self, frame):
        class_ids = []
        confidences = []
        boxes = []
        frame
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255, (self.HEIGHT, self.WIDTH), [0, 0, 0], 1, crop=False
        )
        # Sets the input to the network.
        self.net.setInput(blob)
        # Run the forward pass to get output of the output layers.
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        rows = outputs[0].shape[1]
        image_height, image_width = frame.shape[:2]

        x_factor = image_width / self.WIDTH
        y_factor = image_height / self.HEIGHT
        # 像素中心点
        cx = 0
        cy = 0
        # 循环检测
        res = []
        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            if confidence > self.CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]
                class_id = np.argmax(classes_scores)
                if classes_scores[class_id] > self.SCORE_THRESHOLD:
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    left = int((cx - w / 2) * x_factor)
                    top = int((cy - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

                    
                    indices = cv2.dnn.NMSBoxes(
                        boxes,
                        confidences,
                        self.CONFIDENCE_THRESHOLD,
                        self.NMS_THRESHOLD,
                    )

                    for i in indices:
                        box = boxes[i]
                        left = box[0]
                        top = box[1]
                        width = box[2]
                        height = box[3]
                        
                        cx = left + (width) // 2
                        cy = top + (height) // 2
                        label = "{}:{:.2f}".format(
                            self.classes[class_ids[i]], confidences[i]
                        )
                        self.show_list=[frame,label, cx, cy, left, top,width,height]
                        self.show(self.show_list)
                        result=[ self.classes[class_ids[i]],cx,cy]
                        return result
                      

    def show(self,show_result):
        cv2.circle(show_result[0], (show_result[2],show_result[3]), 5, self.BLUE, 10)
        text_size = cv2.getTextSize(
            show_result[1], self.FONT_FACE, self.FONT_SCALE, self.THICKNESS
        )
        dim, baseline = text_size[0], text_size[1]
        cv2.rectangle(
            show_result[0], (show_result[4], show_result[5]), (show_result[4] + dim[0], show_result[5] + dim[1] + baseline), (0, 0, 0), cv2.FILLED
        )
        cv2.rectangle(
                            show_result[0],
                            (show_result[4], show_result[5]),
                            (show_result[4] + show_result[6], show_result[5] + show_result[7]),
                            self.BLUE,
                            3,
                        )


        
        cv2.putText(
            show_result[0],
            show_result[1],
            (show_result[4], show_result[5] + dim[1]),
            self.FONT_FACE,
            self.FONT_SCALE,
            self.YELLOW,
            self.THICKNESS,
        )
        cv2.imshow("result",show_result[0])
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

camera = Camera()
detector =Yolo(model_path='./yolov5s.onnx',class_names_path='./coco.names')
mc= MyCobot320(utils.get_port_list()[0])
mc.set_tool_reference([0,0,190,0,0,0])
mc.set_end_type(1)
mc.set_pro_gripper_open(14)
time.sleep(2)
T_cam_to_tool = np.array([[-0.6133039050687764, -0.7888530044388438, -0.03961385382918081, -41.45103098720348],
                        [0.7748418721628141, -0.6106250693657994, 0.16357596952253783, -82.48620216354004],
                        [-0.15322660725414114, 0.06962730821899694, 0.9857350783955909, -131.96692630461692],
                        [0.0, 0.0, 0.0, 1.0]])
end=[[84.28, 15.64, -96.15, -11.95, 88.15, 48.51],
[44.82, -53.17, -6.32, -35.41, 92.46, 6.32],
[-29.88, -62.66, 16.34, -35.15, 84.19, 108.45],
[-47.46, -55.89, 26.19, -59.67, 83.75, 93.95]]


def go_take_photo_position():
    mc.send_angles([0,0,-90,0,88,48],100)
    time.sleep(2)


def coordinate_transformation(result):
    cam_coords=None
    while cam_coords is None:
          cam_coords=mc.get_coords()          
          time.sleep(0.2)
          mc.set_end_type(0)
          tmp=mc.get_coords()
          time.sleep(1)
          mc.set_end_type(1)
          target=camera.get_camera_coordinate(result[1],result[2],tmp[2])
          pose = cam_coords
          x, y, z = pose[:3]  
          roll, pitch, yaw = pose[3:]     
          R_matrix = tf3d.euler.euler2mat(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw), 'sxyz')
          T_tool_to_base = tf3d.affines.compose([x, y, z], R_matrix, [1, 1, 1])
          target_homogeneous = np.array([target[0], target[1],1,1])
          target=T_tool_to_base @ T_cam_to_tool @ target_homogeneous
          return target,cam_coords
    

def grab(target,cam_coords):
    cam_coords[0]=target[0]
    cam_coords[1]=target[1]
    cam_coords[2]=44          
    mc.send_coords(cam_coords,50)
    time.sleep(2)
    cam_coords[2]=-13          
    mc.send_coords(cam_coords,50)
    time.sleep(2)          
    mc.set_pro_gripper_angle(14,50)
    time.sleep(1)
    cam_coords[2]=44
    mc.send_coords(cam_coords,50)
    time.sleep(2)
    mc.send_angles([0.7, 14.32, -58.97, -51.15, 89.73, 48.6],100)
    time.sleep(2)


def classify_place(result):
    if result[0] =="banana":
        mc.send_angles(end[0],100)
        time.sleep(2)
        mc.set_pro_gripper_open(14)
        time.sleep(1)
    elif result[0] =="clock":
        mc.send_angles(end[1],100)
        time.sleep(2)
        mc.set_pro_gripper_open(14)
        time.sleep(1)
    elif result[0] =="car":
        mc.send_angles(end[2],100)
        time.sleep(2)
        mc.set_pro_gripper_open(14)
        time.sleep(1)
    elif result[0] =="cat":
        mc.send_angles(end[3],100)
        time.sleep(2)
        mc.set_pro_gripper_open(14)
        time.sleep(1)

    time.sleep(2)

    

if __name__ == "__main__":
    target_name=["banana","clock","cat","car"]
    print("ok")
    go_take_photo_position()
    while True:
        image = camera.get_frame()
       
        result=detector.detect(image)
        
        if  result is None:
            cv2.imshow("result",image)
            cv2.waitKey(1)
        else:
            cam_coords=None
            if result[0] in target_name:
                target,cam_coords=coordinate_transformation(result)
                grab(target,cam_coords)
                classify_place(result)
                go_take_photo_position()


            


            
