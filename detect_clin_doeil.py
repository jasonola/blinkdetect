import cv2
import dlib
import math
BLINK_RATIO_THRESHOLD = 5.7

# Pour compter les clignements
COUNTER = 0
TOTAL = 0

# fonctions pour calculer le ratio de clignement


def midpoint(point1, point2):
    return (point1.x + point2.x)/2, (point1.y + point2.y)/2


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_blink_ratio(eye_points, facial_landmarks):

    # Chopper les points du landmark facial
    corner_left = (facial_landmarks.part(eye_points[0]).x,
                   facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x,
                    facial_landmarks.part(eye_points[3]).y)

    center_top = midpoint(facial_landmarks.part(eye_points[1]),
                          facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]),
                             facial_landmarks.part(eye_points[4]))

    # Calculer les distances pour ensuite chopper le ratio
    horizontal_length = euclidean_distance(corner_left, corner_right)
    vertical_length = euclidean_distance(center_top, center_bottom)

    ratio = horizontal_length / vertical_length

    return ratio


# Video live
#cap = cv2.VideoCapture(0)

# Pour une video enregistrée
cap = cv2.VideoCapture("Video.mp4")


# Nom de l'app
cv2.namedWindow('BlinkDetector')

# Detecter les marques de visage
detector = dlib.get_frontal_face_detector()

# detecter les points faciaux
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# les points pour oeil droit et oeil gauche
left_eye_landmarks = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]

while True:
    # lire les frames
    retval, frame = cap.read()

    # si y a plus de frames, donc video finie, stop et affichage du total
    if not retval:
        print(f"Video terminée, il y'a eu {TOTAL} clignements")

        break

    # noir et blanc
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces, _, _ = detector.run(image=frame, upsample_num_times=0,
                               adjust_threshold=0.0)

    # compter les blinks
    for face in faces:

        landmarks = predictor(frame, face)

        left_eye_ratio = get_blink_ratio(left_eye_landmarks, landmarks)
        right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
        blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blink_ratio < BLINK_RATIO_THRESHOLD:
            COUNTER += 1
        else:
            # si le clignement est lent, il va compter le blink plusieur fois, alors si le blink prend plus que 3 frames, il le compte 1 fois
            if COUNTER >= 3:
                TOTAL += 1
            COUNTER = 0
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('BlinkDetector', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

# detruire la window quand c est fini
cap.release()
print(f"Nombre de clignements : {TOTAL}")
cv2.destroyAllWindows()
