import sys
import cv2

# 学習済のHaar-like特徴を用いた分類器のデータ
# もしカスケード分類器用のファイルが無ければgithubからダウンロードもできる
# https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade_path = './haarcascades/haarcascade_frontalface_default.xml'
eye_cascade_path = './haarcascades/haarcascade_eye.xml'
smile_cascade_path = './haarcascades/haarcascade_smile.xml'

# 引数(かっこの中の変数)： カスケード分類器の設定ファイル
# カスケード分類器の設定ファイルは自動的に作成される
# カスケードのファイルを読み込み
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

# 画像を読み込み
# imreadもcv2のパッケージの中に含まれている
# 相対パスで画像のファイルの読み込みができる
src = cv2.imread('./test_face.jpg')

# 白黒にする(二次元配列)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# カスケード定義ファイルのデータ精査
faces = face_cascade.detectMultiScale(src_gray)
smile = smile_cascade.detectMultiScale(src_gray)

# カスケードファイルの中身が無かったらエラーを吐く
if(len(faces) == 0):
    print("Failed")
    # プログラムを終了させる
    sys.exit()

# 検出した箇所を枠線で囲むためにループ
for x, y, w, h in faces:
    # 検出された顔を枠線で囲む(青色)
    cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # 検出された顔を(三次元配列)をfaceに格納(多分)
    face = src[y: y + h, x: x + w]
    # 白黒にした写真(二次元配列)をface_grayに格納(多分)
    face_gray = src_gray[y: y + h, x: x + w]
    # face_grayから目を検出
    eyes = eye_cascade.detectMultiScale(face_gray)
    # 検出した目を枠線で囲む
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


# 画像の書き込み(新しい名前にする→上書きされないように)
cv2.imwrite('./test_write.jpg', src)
