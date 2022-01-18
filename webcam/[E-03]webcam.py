import os
import cv2
import platform
import dlib
import copy
import time
import sys

import matplotlib.pyplot as plt
import numpy as np

def main():
    #Jupyter
    #local_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper"
    sticker_path = os.path.join("{}".format(slashs).join(local_path.split(slashs)[:-1]), 'images', 'cat_mustache.png')
    img_sticker = cv2.imread(sticker_path)

    ######################################## 얼굴 검출로 HOG, SVM사용
    detector_hog = dlib.get_frontal_face_detector()

    ######################################## 이목구비의 위치를 추론(face landmark localization)
    model_path = os.path.join("{}".format(slashs).join(local_path.split(slashs)[:-1]), 'models', 'shape_predictor_68_face_landmarks.dat')
    landmark_predictor = dlib.shape_predictor(model_path)

    cap = cv2.VideoCapture(0)
    print('width :%d, height : %d' % (cap.get(3), cap.get(4)))

    while True:
        ret, frame = cap.read()    # Read 결과와 frame

        if(ret):
            frame = cv2.cvtColor(frame,  cv2.COLOR_BGR2RGB)

            dlib_rects = detector_hog(frame, 1)
            print(dlib_rects)
            if len(dlib_rects)>0:
                for dlib_rect in dlib_rects:
                    l = dlib_rect.left()
                    t = dlib_rect.top()
                    r = dlib_rect.right()
                    b = dlib_rect.bottom()
                    cv2.rectangle(frame, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)

                points = landmark_predictor(frame, dlib_rects[0])
                img_sticker_copy = copy.deepcopy(img_sticker)
                list_points = list(map(lambda p: (p.x, p.y), points.parts()))

                grad = np.arctan(-(list_points[16][1] - list_points[0][1]) / (list_points[16][0] - list_points[0][0]))
                x = list_points[33][0]
                y = (list_points[30][1] + list_points[33][1]) // 2
                w = h = int(dlib_rects[0].width()*1.5)

                img_sticker_copy = cv2.resize(img_sticker_copy, (w, h))
                angle_m = cv2.getRotationMatrix2D((int(w/2),int(h/2)),np.rad2deg(grad),1)
                img_sticker_copy = cv2.warpAffine(img_sticker_copy, angle_m,(img_sticker_copy.shape[0], img_sticker_copy.shape[1]))

                refined_x = x - w // 2
                refined_y = y - h // 2
                if refined_x < 0: 
                    img_sticker_copy = img_sticker_copy[:, -refined_x:]
                    refined_x = 0
                if refined_y < 0:
                    img_sticker_copy = img_sticker_copy[-refined_y:, :]
                    refined_y = 0

                sticker_area = frame[refined_y:refined_y+img_sticker_copy.shape[0], refined_x:refined_x+img_sticker_copy.shape[1]]
                frame[refined_y:refined_y+img_sticker_copy.shape[0], refined_x:refined_x+img_sticker_copy.shape[1]] = \
                        np.where(img_sticker_copy==0,sticker_area,img_sticker_copy).astype(np.uint8)

            cv2.imshow('frame_color', cv2.cvtColor(frame,  cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    local_path = os.getcwd()
    slashs = '\\' if platform.system() == 'Windows' else '/'
    main()