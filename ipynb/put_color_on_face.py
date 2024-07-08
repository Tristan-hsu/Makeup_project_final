import cv2
import mediapipe as mp
import numpy as np

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

cv2.namedWindow('Trackbars')
cv2.createTrackbar('R', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('G', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('B', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('level', 'Trackbars', 0, 100, nothing)
cv2.createTrackbar('region_rate', 'Trackbars', 0, 100, nothing)

# Path to the image
# image_path = './256img_lst/006.jpg'
# rwmask_path ='./pymatting_outcome_rw/006.jpg'

def put_color_on_face(image, mask):


    # Read the image
    image = cv2.resize(image, (128, 128))
    mask = cv2.resize(mask, (128, 128))

    if image is None:
        print(f"Failed to load image: {image_path}")
        exit()

    if mask is None:
        print(f"Failed to load image: {rwmask_path}")
        exit()

    # Detect face in the image
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = image.shape
            landmarks = [(int(point.x * width), int(point.y * height)) for point in face_landmarks.landmark]

    while True:
        # Get the current positions of the trackbars
        r = cv2.getTrackbarPos('R', 'Trackbars')
        g = cv2.getTrackbarPos('G', 'Trackbars')
        b = cv2.getTrackbarPos('B', 'Trackbars')
        const = cv2.getTrackbarPos('level', 'Trackbars')
        region_rate = cv2.getTrackbarPos('region_rate', 'Trackbars')


        const = -(float(const)-50)/500+0.9
        img = np.zeros((100, 100, 3), np.uint8) 
        img[:] = [b,g,r]
        # Apply the makeup effect with the current color
        if results.multi_face_landmarks:
            output_image = apply_eyeshade(image.copy(),rwmask.copy() , (b, g, r),region_rate,const)
        else:
            output_image = image.copy()
        output_image = cv2.resize(output_image, (512, 512))
        # Display the result
        cv2.imshow('Lipstick Application', output_image)
        cv2.imshow('color plate',img)

        # Exit on 'esc' key press
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    face_mesh.close()

image_path = './origin.jpeg'
rwmask_path ='./outcome.jpg'