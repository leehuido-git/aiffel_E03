import os
import cv2
import platform
import dlib
import copy
import time
import sys

import matplotlib.pyplot as plt
import numpy as np

check = False

def load_path(img_dir):
    ext = ['.JPG', '.jpg', '.png', '.PNG']
    img_paths = []

    for (path, dir, files) in os.walk(img_dir):
        for filename in files:
            if os.path.splitext(filename)[-1] in ext:
                img_paths.append(os.path.join(path, filename))
    print("이미지 개수는 {}입니다.".format(len(img_paths)))
    if len(img_paths) == 0:
        print("이미지가 없습니다!")
        sys.exit()
    return img_paths

def main():
    #Jupyter
    #local_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper"
    img_paths = load_path(os.path.join(local_path, 'input'))

    sticker_path = os.path.join("{}".format(slashs).join(local_path.split(slashs)[:-1]), 'images', 'cat_mustache.png')
    img_sticker = cv2.imread(sticker_path)

    ######################################## 얼굴 검출로 HOG, SVM사용
    detector_hog = dlib.get_frontal_face_detector()

    ######################################## 이목구비의 위치를 추론(face landmark localization)
    model_path = os.path.join("{}".format(slashs).join(local_path.split(slashs)[:-1]), 'models', 'shape_predictor_68_face_landmarks.dat')
    landmark_predictor = dlib.shape_predictor(model_path)

    start_time = time.time()
    for img_path in img_paths:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)    #한글경로 인식문제
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_show = copy.deepcopy(img)
        ######################################## 이미지 확인
        if check:
            plt.imshow(img_show)        
            plt.show()

        ######################################## 얼굴 검출로 HOG, SVM사용
        dlib_rects = detector_hog(img, 1)
        """
        dlib detector 는 dlib.rectangles 타입의 객체를 반환합니다. dlib.rectangles 는 dlib.rectangle 객체의 배열 형태로 이루어져 있습니다.
        dlib.rectangle객체는 left(), top(), right(), bottom(), height(), width() 등의 멤버 함수를 포함하고 있습니다. 
        """
        ######################################## 이미지 확인
        if check:
            for dlib_rect in dlib_rects:
                l = dlib_rect.left()
                t = dlib_rect.top()
                r = dlib_rect.right()
                b = dlib_rect.bottom()
                cv2.rectangle(img_show, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)
            plt.imshow(img_show)
            plt.show()

        ######################################## 이목구비의 위치를 추론(face landmark localization)
        ######################################## top-down방식: bounding box를 찾고 box 내부의 keypoint를 예측
        ######################################## ibug 300w 데이터셋(68개 keypoint), regression tree의 앙상블 모델사용
        list_landmarks = []
        for dlib_rect in dlib_rects:
            points = landmark_predictor(img, dlib_rect)
            list_points = list(map(lambda p: (p.x, p.y), points.parts()))
            list_landmarks.append(list_points)

        ######################################## 이미지 확인
        if check:
            for landmark in list_landmarks:
                for point in landmark:
                    cv2.circle(img_show, point, 2, (0, 255, 255), -1)
            plt.imshow(img_show)
            plt.show()

        ######################################## 스티커 적용하기
        for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
            grad = np.arctan(-(landmark[16][1] - landmark[0][1]) / (landmark[16][0] - landmark[0][0]))
            x = landmark[33][0]
            y = (landmark[30][1] + landmark[33][1]) // 2
            w = h = int(dlib_rect.width()*1.5)

        img_sticker = cv2.resize(img_sticker, (w, h))
        angle_m = cv2.getRotationMatrix2D((img_sticker.shape[0]/2,img_sticker.shape[1]/2),np.rad2deg(grad),1) 
        img_sticker = cv2.warpAffine(img_sticker, angle_m,(img_sticker.shape[0], img_sticker.shape[1]))
        refined_x = x - w // 2
        refined_y = y - h // 2
        if refined_x < 0: 
            img_sticker = img_sticker[:, -refined_x:]
            refined_x = 0
        if refined_y < 0:
            img_sticker = img_sticker[-refined_y:, :]
            refined_y = 0

        sticker_area = img[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
        img[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
                np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)
        if check:
            sticker_area = img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
            img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
                    np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)
            plt.imshow(img_show)
            plt.show()
        print("{:.5f}".format(time.time()-start_time))

        save_path_split = img_path.split(slashs)
        save_path_split[-2] = 'result'
        plt.imshow(img)
        plt.savefig("{}".format(slashs).join(save_path_split))
        if check:
            plt.show()

    print("한장당 평균 {:.5f}:sec".format((time.time()-start_time)/len(img_paths)))



if __name__ == '__main__':
    local_path = os.getcwd()
    slashs = '\\' if platform.system() == 'Windows' else '/'
    main()