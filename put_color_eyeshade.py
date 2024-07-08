import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Face Mesh and drawing utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Define the makeup effect function
def apply_eyeshade(image, mask, color,region_rate ,const = 0.9 ):
    
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask1 = np.all(mask >255-region_rate*255/100, axis=-1)

    mask2 = image.copy()
    mask2[mask1] = gray_mask[mask1][:, np.newaxis] *color/255

# Parameters for the GuidedFilter
    radius = 10         # Radius of the guided filter
    eps = 0.1         # Regularization parameter (epsilon)

    guided_filter = cv2.ximgproc.createGuidedFilter(guide=gray_mask, radius=radius, eps=eps)
    filtered_mask = guided_filter.filter(mask2)
    result = cv2.addWeighted(image, const, filtered_mask, 1-const, 0)

    return result

# Create trackbars to adjust lip color
def nothing(x):
    pass

def create_trackbar():
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('R', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('G', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('B', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('level', 'Trackbars', 0, 100, nothing)
    cv2.createTrackbar('region_rate', 'Trackbars', 0, 100, nothing)

def get_trackbar():
    r = cv2.getTrackbarPos('R', 'Trackbars')
    g = cv2.getTrackbarPos('G', 'Trackbars')
    b = cv2.getTrackbarPos('B', 'Trackbars')
    const = cv2.getTrackbarPos('level', 'Trackbars')
    region_rate = cv2.getTrackbarPos('region_rate', 'Trackbars')
    return r,g,b,const,region_rate

def read_and_resize(image_path,mask_path):
    # Read the image
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        exit()
    if mask is None:
        print(f"Failed to load image: {mask_path}")
        exit()
    image = cv2.resize(image, (128, 128))
    mask = cv2.resize(mask, (128, 128))
    return image, mask

def show_img(output_image,r,g,b):
    img = np.zeros((100, 100, 3), np.uint8) 
    img[:] = [b,g,r]
    cv2.imshow('Lipstick Application', output_image)
    cv2.imshow('color plate',img)

def put_color_on_face(image, mask,r=0,g=0,b=255,const=100,region_rate=50):
    # create_trackbar()
    # Detect face in the image
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = image.shape
            landmarks = [(int(point.x * width), int(point.y * height)) for point in face_landmarks.landmark]
    # while True:
    #     # Get the current positions of the trackbars
    #     # r,g,b,const,region_rate = get_trackbar()
        
    #     # const compute method
    #     const = -(float(const)-50)/500+0.9

    #     # Apply the makeup effect with the current color
    #     if results.multi_face_landmarks:
    #         output_image = apply_eyeshade(image.copy(),mask.copy() , (b, g, r),region_rate,const)
    #     else:
    #         output_image = image.copy()
        
    #     output_image = cv2.resize(output_image, (512, 512))
    #     # Display the result
    #     show_img(output_image,r,g,b)
    #     # Exit on 'esc' key press
    #     if cv2.waitKey(1) & 0xFF == 27:
    #         break
    const = -(float(const)-50)/500+0.9
    output_image = apply_eyeshade(image.copy(),mask.copy() , (b, g, r),region_rate,const)
    output_image = cv2.resize(output_image, (512, 512))
    # show_img(output_image,r,g,b)
    # cv2.waitKey(0)
    
    return output_image

def put_color_on_face_with_bar(image, mask):
    create_trackbar()
    # Detect face in the image
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = image.shape
            landmarks = [(int(point.x * width), int(point.y * height)) for point in face_landmarks.landmark]
    while True:
        # Get the current positions of the trackbars
        r,g,b,const,region_rate = get_trackbar()
        
        # const compute method
        const = -(float(const)-50)/500+0.9

        # Apply the makeup effect with the current color
        if results.multi_face_landmarks:
            output_image = apply_eyeshade(image.copy(),mask.copy() , (b, g, r),region_rate,const)
        else:
            output_image = image.copy()
        
        output_image = cv2.resize(output_image, (512, 512))
        # Display the result
        show_img(output_image,r,g,b)
        # Exit on 'esc' key press
        if cv2.waitKey(1) & 0xFF == 27:
            break

if __name__ == '__main__':
    image_path = 'C:/Users/User/makeup_project_finaltest/cut_face'
    mask_path ='C:/Users/User/makeup_project_finaltest/predict'
    final_path = 'C:/Users/User/makeup_project_finaltest/final'
    for filename in os.listdir(image_path):
        image, mask = read_and_resize(image_path+'/'+filename,mask_path+'/'+filename)
        r,g,b,const,region_rate= 0,0,200,100,30
        output_image=put_color_on_face(image, mask,r,g,b,const,region_rate)
        cv2.imwrite(final_path+'/'+filename,output_image)
    # put_color_on_face_with_bar(image, mask)
    face_mesh.close()



