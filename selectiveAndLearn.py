# -*- coding: UTF-8 -*-
# for selective search
import cv2
import selectivesearch

#for ResNet50
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# for VGG16
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import sys

def predictFromVGG16(img):
    # 学習済みのVGG16をロード
    # 構造とともに学習済みの重みも読み込まれる
    model = VGG16(weights='imagenet')
    # model.summary()
    
    # 引数で指定した画像ファイルを読み込む
    # サイズはVGG16のデフォルトである224x224にリサイズされる
    # img = image.load_img(filename, target_size=(224, 224))
    
    # 読み込んだPIL形式の画像をarrayに変換
    x = image.img_to_array(img)
    
    # 3次元テンソル（rows, cols, channels) を
    # 4次元テンソル (samples, rows, cols, channels) に変換
    # 入力画像は1枚なのでsamples=1でよい
    x = np.expand_dims(x, axis=0)
    
    # Top-5のクラスを予測する
    # VGG16の1000クラスはdecode_predictions()で文字列に変換される
    preds = model.predict(preprocess_input(x))
    results = decode_predictions(preds, top=1)[0]
    obj = []
    for result in results:
        # 20%以上で予測されたものに限定する
        # 物が検知されなくても表示するため低くしている
        # 上げるとちゃんと検知されないと何も表示されない
        if result[2] > 0.2:
            print(result)
            obj = (result[1], result[2])
        else:
            obj = (-1, -1)
        break
    return obj



cv2.namedWindow('interactive-camera') # create win with win name

WIN_WIDTH = 320
WIN_HEIGHT = 180

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_HEIGHT)

def main():
    while(True):
 
        # 動画ストリームからフレームを取得
        ret, frame = cap.read()
        img = frame

        # perform selective search
        img_lbl, regions = selectivesearch.selective_search(
            img, scale=500, sigma=0.9, min_size=10)
        
        candidates = set()
        
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding regions smaller than 200 pixels
            if r['size'] < 2000:
                continue
            # distorted rects
            x, y, w, h = r['rect']
            if w / h > 1.2 or h / w > 1.2:
                continue
            candidates.add(r['rect'])
        
        #画像への枠作成
        for region in candidates:
            x,y,w,h = region
            
            # 物体っぽい場所をトリミングする
            trimmed_img = img[x:x+w, y:y+h]
            
            if len(trimmed_img) == 0:
                break
            
            # 画像を224*224にする
            trimmed_img = cv2.resize(trimmed_img, (224, 224))
            
            # トリミングされた画像が何であるか予測（入力はカラー画像）
            object_name, proba = predictFromVGG16(trimmed_img)
            
            # (object_name, proba) == (-1, -1)の時オブジェクトが無い
            if object_name is not -1:
                # トリミングされた場所のマーキング
                color = (100, 200, 100)
                cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), thickness=2)
                cv2.putText(img, '{0} {1:.2}%'.format(object_name, proba*100), (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), thickness=1)
        
        cv2.imshow("camera window", img) 
        
        # escを押したら終了。
        if cv2.waitKey(1) == 27:
            break
    
    #終了
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    main()