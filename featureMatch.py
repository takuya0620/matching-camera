# https://qiita.com/hitomatagi/items/caac014b7ab246faf6b1
# http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
# http://www.cellstat.net/homography/

# 教師画像のリサイズ, 特徴点検出の重み, 特徴点の最低検出数でうまくいくか変わる

# -*- coding: utf-8 -*-
import cv2
import numpy as np

def detect_item(img2):
    # 画像１
    img1 = cv2.imread("orig.jpg", 0)
    # 大きすぎる場合はリサイズで軽量化
    img1 = cv2.resize(img1,(200, 150))
    
    # 画像２
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(img2,(200, 150))
    
    
    # A-KAZE検出器の生成
    akaze = cv2.AKAZE_create()                                
    
    # 特徴量の検出と特徴量ベクトルの計算
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    
    # Brute-Force Matcher生成
    bf = cv2.BFMatcher()
    
    # 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
    matches = bf.knnMatch(des1, des2, k=2)
    
    # データを間引きする
    ratio = 0.8
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])
    
    #教師画像と同じ図形を四角で囲む(要検討)
    MIN_MATCH_COUNT=5
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,2.0)
        matchesMask = mask.ravel().tolist()
        
        if M is not None:
            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
        else:
            dst = []
        
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    
    img2 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:30], None, flags = 2)
    return img2




cv2.namedWindow('interactive-camera') # create win with win name

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while(True):

    ret, frame = cap.read()
    if not ret: continue
    
    frame = detect_item(frame)
    
    cv2.imshow('match-camera', frame)  # show in the win

    k = cv2.waitKey(1)
    if k == ord('q') or k == 27:
        break

cap.release()
cv2.destroyAllWindows()