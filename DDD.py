#!/usr/bin/env python
from genericpath import isdir
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import math
import cv2
import numpy as np
from sympy import capture
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
from pathlib import Path
from IsDrows import isDrows

# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    '/Users/choehyeonbin/face_landmark/Driver-Drowsiness-Detection/dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] initializing camera...")

# vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
# time.sleep(2.0)

# 400x225 to 1024x576
frame_width = 1024
frame_height = 576

# loop over the frames from the video stream
# 2D image points. If you change the image, you need to change vector
image_points = np.array([
    (359, 391),     # Nose tip 34
    (399, 561),     # Chin 9
    (337, 297),     # Left eye left corner 37
    (513, 301),     # Right eye right corne 46
    (345, 465),     # Left Mouth corner 49
    (453, 469)      # Right mouth corner 55
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 30            # 무조건 잔다고 판단 할 수 있는 프레임 수
COUNTER = 0
UNDET_FPS = 0
DT_LOOK_FPS = 0
EYE_SCORE = 0

head_tilt_degree_x, head_tilt_degree_z = 0, 0

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)
save_path = "/Users/choehyeonbin/face_landmark/headpose_origin/runs"
# vid_writer = cv2.VideoWriter(f'{save_path}/dst.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))
# vs = cv2.VideoCapture("headpose_origin/none.mp4")
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it toP
    # grayscale
    # frame = vs.read()
    _, frame = vs.read()
    frame = imutils.resize(frame, width=1024, height=576)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # check to see if a face was detected, and if so, draw the total
    # number of faces on the frame
    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        UNDET_FPS = 0
    else:
        UNDET_FPS += 1

    # loop over the face detections
    for rect in rects:
        # compute the bounding box of the face and draw it on the
        # frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)      # 얼굴 박스그리기
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            # if the eyes were closed for a sufficient number of times
            # then show the warning
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Eyes Closed!", (500, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
        else:
            COUNTER = 0
        
        EYE_SCORE = int(COUNTER/EYE_AR_CONSEC_FRAMES * 130)


        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw each of them
        for (i, (x, y)) in enumerate(shape):
            if i == 30:                                                             # 코 끝 번호
                image_points[0] = np.array([x, y], dtype='double')
            elif i == 8:
                image_points[1] = np.array([x, y], dtype='double')
            elif i == 36:
                image_points[2] = np.array([x, y], dtype='double')
            elif i == 45:
                image_points[3] = np.array([x, y], dtype='double')
            elif i == 48:
                image_points[4] = np.array([x, y], dtype='double')
            elif i == 54:
                image_points[5] = np.array([x, y], dtype='double')

        # head_tilt_degree_z z축 각도, head_tilt_degree_x x축 각도
        (head_tilt_degree_z, start_point, end_point, 
            end_point_alt, end_point_alt1, head_tilt_degree_x) = getHeadTiltAndCoords(size, image_points, frame_height, frame_width)
        
        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)
        cv2.line(frame, start_point, end_point_alt1, (0, 255, 0), 2)

        if head_tilt_degree_z:
            cv2.putText(frame, f'Head Tilt Degree_Z: {head_tilt_degree_z[0]:.2f}', (170, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f'Head Tilt Degree_x: {head_tilt_degree_x[0]:.2f}', (170, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # extract the mouth coordinates, then use the
        # coordinates to compute the mouth aspect ratio
    # show the frameq
    score_x = abs(head_tilt_degree_x)/35 * 100
    if head_tilt_degree_z >= 180:
        score_z = abs(head_tilt_degree_z-360)/15 *100
    else:
        score_z = head_tilt_degree_z/15 * 100
    
    status = isDrows(EYE_SCORE, [score_x, score_z], UNDET_FPS)
    if status == 1:
        cv2.putText(frame, "Drowsines!", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 5)
    elif status == 2:
        DT_LOOK_FPS+=1
        if DT_LOOK_FPS > EYE_AR_CONSEC_FRAMES:
            cv2.putText(frame, "Look Forward!", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 5)
        elif DT_LOOK_FPS < EYE_AR_CONSEC_FRAMES:
            cv2.putText(frame, "CAUTION!", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,255), 5)
    else:
        DT_LOOK_FPS = 0
    vid_writer.write(frame)                       # 주석 제거 시 동영상 저장 저장경로는 line 59
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        vid_writer.release()
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

