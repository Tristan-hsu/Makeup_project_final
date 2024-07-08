import cv2
import numpy as np

# 使用 OpenCV 加載預訓練的人臉檢測器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def correct_and_crop_face(image):
    # 轉換成灰度圖像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 檢測人臉
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # 校正人臉角度並裁剪
        # rotated_img, (x, y, w, h) = correct_face_orientation(gray, image, faces)

        # 裁剪人臉
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            h, w = face.shape[:2]
            diff = abs(h - w) // 2
            if h > w:
                face = face[diff:h-diff, :]
            else:
                face = face[:, diff:w-diff]

        # 調整大小為256x256
        resized_face = cv2.resize(face, (256, 256))
        return resized_face

    else:
        print(f"No face detected in the provided image")
        return None

# def correct_face_orientation(gray, img, faces):
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            # 取前兩個眼睛
            eye1 = eyes[0]
            eye2 = eyes[1]
            
            if eye1[0] > eye2[0]:
                eye1, eye2 = eye2, eye1
            
            left_eye_center = (int(eye1[0] + eye1[2] / 2), int(eye1[1] + eye1[3] / 2))
            right_eye_center = (int(eye2[0] + eye2[2] / 2), int(eye2[1] + eye2[3] / 2))

            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dy, dx))

            # 旋轉圖片
            eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                           (left_eye_center[1] + right_eye_center[1]) // 2)
            M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
            rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            return rotated, (x, y, w, h)

    return img, (x, y, w, h)

def read_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    return image


# 執行示例
# 讀取圖片
# input_image_path = 'path_to_your_image.jpg'  # 替換成您的圖片路徑
# img = cv2.imread(input_image_path)
# if img is None:
#     print(f"Error: Unable to read image at {input_image_path}")
# else:
#     process_single_image(img)

if __name__ == '__main__':
    folder = 'C:/Users/User/makeup_project_finaltest/oringinal_photo'
    filename = 'ori.jpg'
    output_path='C:/Users/User/makeup_project_finaltest/cut_face'
    outputfilename = "cut.jpg"
    image = read_image(folder + '/'+ filename)
    
    corrected_face = correct_and_crop_face(image)
    cv2.imwrite(output_path+'/'+outputfilename,corrected_face)
