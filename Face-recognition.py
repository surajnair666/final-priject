
import cv2
import os
import numpy as np

subjects = ["", "Suraj", "Roshan"]



#function to detect face using OpenCV
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]

    return gray[y:y+w, x:x+h], faces[0]



def prepare_training_data(data_folder_path):


    dirs = os.listdir(data_folder_path)


    faces = []
    labels = []

    for dir_name in dirs:


        if not dir_name.startswith("s"):
            continue;

        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name

        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:

            if image_name.startswith("."):
                continue;

            image_path = subject_dir_path + "/" + image_name

            image = cv2.imread(image_path)

            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)

            face, rect = detect_face(image)


            if face is not None:

                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels

print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Done")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))



face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)

    label, confidence = face_recognizer.predict(face)
    label_text = subjects[label]

    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)

    return img

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables)
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        prediction = model.predict(face_resize) 
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1]<500:

           cv2.putText(im, '% s - %.0f' %
(names[prediction[0]], prediction[1]), (x-10, y-10),
cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        else:
          cv2.putText(im, 'not recognized',
(x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    cv2.imshow('OpenCV', im)

    key = cv2.waitKey(10)
    if key == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
