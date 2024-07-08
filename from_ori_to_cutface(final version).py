#將臉部方形反轉顏色
import cv2
import numpy as np
import mediapipe as mp
import os

# 初始化MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
def cut_mp256(image):
# 初始化MP窗口
    cv2.namedWindow('MediaPipe Face Mesh', cv2.WINDOW_NORMAL)

    h, w, _ = image.shape
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 提取所有关键点的坐标
            x_min = w
            y_min = h
            x_max = y_max = 0
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min: x_min = x
                if y < y_min: y_min = y
                if x > x_max: x_max = x
                if y > y_max: y_max = y
            # 计算正方形区域
            face_width = x_max - x_min
            face_height = y_max - y_min
            face_size = max(face_width, face_height)
            center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
            # 确保正方形区域不超出边界
            left = max(center_x - face_size // 2, 0)
            right = min(center_x + face_size // 2, w)
            top = max(center_y - face_size // 2, 0)
            bottom = min(center_y + face_size // 2, h)
            # 裁剪人脸区域并调整大小
            face_crop = image[top:bottom, left:right]
            face_crop_resized = cv2.resize(face_crop, (256, 256))
            #=====================================================傳回256*256
            
            # 将反转后的图像调整回原始大小
            # face_crop_inverted = cv2.resize(face_crop_resized, (right-left, bottom-top))
            # # 将反转后的图像粘贴回原始图像中
            # image[top:bottom, left:right] = face_crop_inverted
            # # 绘制边界框
            # # cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    return face_crop_resized,top,bottom,left,right
#使用image,top,bottom,left,right來接值
#或是image,_,_,_,_來接
if __name__== "__main__":
    input_path='C:/Users/User/makeup_project_finaltest/oringinal_photo/'
    output_path='C:/Users/User/makeup_project_finaltest/cut_face'
    for filename in os.listdir(input_path):
        image=cv2.imread(input_path+'/'+filename)
        image,_,_,_,_=cut_mp256(image)
        cv2.imwrite(output_path+'/'+filename,image)


    # cv2.imshow('MediaPipe Face Mesh', image)
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # # 释放所有窗口
    # cv2.destroyAllWindows()
