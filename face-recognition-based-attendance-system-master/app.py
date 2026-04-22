import cv2
import os
import time
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Defining Flask App
app = Flask(__name__)

nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

attendance_file = f'Attendance/Attendance-{datetoday}.csv'
if not os.path.isfile(attendance_file) or os.stat(attendance_file).st_size == 0:
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time')

# Global model variable
model = None


def load_trained_model():
    global model
    if os.path.isdir('static') and 'face_recognition_model.pkl' in os.listdir('static'):
        try:
            model = joblib.load('static/face_recognition_model.pkl')
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            model = None
    else:
        logging.warning(
            "No face_recognition_model.pkl found in static folder.")
        model = None


# Load model at startup
load_trained_model()


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # scaleFactor=1.3 is faster than 1.2. minNeighbors=5 is standard.
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    except Exception as e:
        logging.error(f"Error in extract_faces: {e}")
        return []


# Identify face using ML model
def identify_face(facearray):
    if model is None:
        load_trained_model()
    if model:
        return model.predict(facearray)
    return ["Unknown"]


# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            if img is not None:
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face.ravel())
                labels.append(user)

    if not faces:
        logging.warning("No training data found.")
        return

    faces = np.array(faces)

    # Determine safe number of components for PCA
    n_samples = faces.shape[0]
    n_features = faces.shape[1]
    n_components = min(n_samples, n_features, 50)

    logging.info(
        f"Training model with {n_samples} samples and PCA(n_components={n_components})")

    # Create a pipeline with Scaling, PCA, and KNN
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])

    pipeline.fit(faces, labels)

    # Save model with error handling
    try:
        model_path = 'static/face_recognition_model.pkl'
        joblib.dump(pipeline, model_path)
        logging.info(f"Model saved successfully at {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        return

    # Update global model
    global model
    model = pipeline
    logging.info("Model training completed and updated.")


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    attendance_file = f'Attendance/Attendance-{datetoday}.csv'
    if not os.path.isfile(attendance_file) or os.stat(attendance_file).st_size == 0:
        with open(attendance_file, 'w') as f:
            f.write('Name,Roll,Time')

    df = pd.read_csv(attendance_file)
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    attendance_file = f'Attendance/Attendance-{datetoday}.csv'
    with open(attendance_file, 'a') as f:
        f.write(f'\n{username},{userid},{current_time}')


# A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


# A function to delete a user folder
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)


################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)

    # if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/') == []:
        os.remove('static/face_recognition_model.pkl')

    try:
        train_model()
    except:
        pass

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# Our main Face Recognition functionality.
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    global datetoday, datetoday2
    names, rolls, times, l = extract_attendance()

    if model is None:
        load_trained_model()

    if model is None:
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    # Pre-load attendance to avoid CSV reading in loop
    attendance_file = f'Attendance/Attendance-{datetoday}.csv'
    if not os.path.isfile(attendance_file) or os.stat(attendance_file).st_size == 0:
        with open(attendance_file, 'w') as f:
            f.write('Name,Roll,Time')

    df = pd.read_csv(attendance_file)
    already_marked = set(df['Roll'].astype(int))

    cap = cv2.VideoCapture(0)
    logging.info("Attendance tracking started.")

    start_time = time.time()
    timeout_duration = 5  # Automatic stop after 60 seconds

    while True:
        # Check for timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_duration:
            logging.info(
                f"Attendance tracking auto-stopped after {timeout_duration} seconds")
            break

        # Update datetoday and datetoday2 in case of date change
        dt = datetime.now()
        datetoday = dt.strftime("%m_%d_%y")
        datetoday2 = dt.strftime("%d-%B-%Y")

        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for better face detection
        h, w = frame.shape[:2]
        target_w = 640
        scale = target_w / w
        target_h = int(h * scale)
        small_frame = cv2.resize(frame, (target_w, target_h))

        # Detect faces on every frame for multiple person support
        faces_raw = extract_faces(small_frame)

        # Scale coordinates back to original frame size
        faces = []
        for (x, y, w_face, h_face) in faces_raw:
            faces.append((int(x/scale), int(y/scale),
                         int(w_face/scale), int(h_face/scale)))

        # Display remaining time
        remaining_time = int(timeout_duration - elapsed_time)

        # Process ALL detected faces
        for (x, y, w, h) in faces:
            try:
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]

                # Default UI state: Person identified but not yet marked
                rect_color = (86, 32, 251)  # Purple-ish
                label_text = f'{identified_person}'

                # Mark attendance for all identified people
                if identified_person != "Unknown" and '_' in identified_person:
                    try:
                        userid = int(identified_person.split('_')[1])
                        if userid not in already_marked:
                            add_attendance(identified_person)
                            already_marked.add(userid)
                            logging.info(
                                f"Attendance MARKED for: {identified_person}")
                            rect_color = (39, 174, 96)  # Green (Success)
                            label_text = f'{identified_person} (MARKED)'
                        else:
                            # Orange (Already marked)
                            rect_color = (255, 165, 0)
                            label_text = f'{identified_person} (Present)'
                    except (IndexError, ValueError) as e:
                        logging.error(
                            f"Error parsing identified person '{identified_person}': {e}")

                cv2.rectangle(frame, (x, y), (x+w, y+h), rect_color, 2)
                cv2.rectangle(frame, (x, y), (x+w, y-40), rect_color, -1)
                cv2.putText(frame, label_text, (x+5, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            except Exception as e:
                logging.error(f"Error processing face: {e}")
                continue

        # Display countdown timer
        cv2.putText(frame, f'Time Remaining: {remaining_time}s', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Attendance', frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        if len(faces) > 0:
            # Only process the first (largest) face to avoid multiple users
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1

        if i >= nimgs:
            break

        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
