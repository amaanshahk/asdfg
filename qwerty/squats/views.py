# squats/views.py
from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def count_rep(counter, stage, angle, up_threshold, down_threshold):
    if angle > up_threshold:
        stage = "up"
    if angle < down_threshold and stage == 'up':
        stage = "down"
        counter += 1
        print("Rep Count:", counter)
    return counter, stage

def generate_frames():
    cap = cv2.VideoCapture(0)
    squat_counter = 0
    squat_stage = None
    squat_up_threshold = 170
    squat_down_threshold = 80

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Squat tracking for knee
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                heel_l = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]

                angle_lk = int(calculate_angle(hip_l, knee_l, heel_l))

                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                heel_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]

                angle_rk = int(calculate_angle(hip_r, knee_r, heel_r))

                avg_knee_angle = (angle_lk + angle_rk) / 2

                # Squat tracking for hip
                angle_lh = int(calculate_angle(shoulder_l, hip_l, knee_l))
                angle_rh = int(calculate_angle(shoulder_r, hip_r, knee_r))
                avg_hip_angle = (angle_lh + angle_rh) / 2

                # Squat counting based on knee and hip angles
                squat_counter, squat_stage = count_rep(squat_counter, squat_stage, avg_knee_angle,
                                                       squat_up_threshold, squat_down_threshold)

                # Visualize angles at knee points
                cv2.putText(image, 'Left Knee Angle: {:.2f}'.format(angle_lk), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, 'Right Knee Angle: {:.2f}'.format(angle_rk), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Visualize angles at hip points
                cv2.putText(image, 'Left Hip Angle: {:.2f}'.format(angle_lh), (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, 'Right Hip Angle: {:.2f}'.format(angle_rh), (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Render squat counter
                cv2.putText(image, 'Squat Reps: ' + str(squat_counter), (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                           mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                           )

            except Exception as e:
                print("Error:", e)

            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

def index(request):
    return render(request, 'squats/index.html')

def video_feed(request):
    return StreamingHttpResponse(generate_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
