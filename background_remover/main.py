import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os


cap = cv2.VideoCapture(0)
cap.set(3, 640)  # 3 for width
cap.set(4, 480)  # 4 for height

segmentor = SelfiSegmentation()

fps_read = cvzone.FPS()

list_background_images_path = os.listdir("images")
background_images = []
for path in list_background_images_path:
    img_background = cv2.imread(f"images/{path}")
    img_background = cv2.resize(img_background, (640, 480))
    background_images.append(img_background)
image_index = 0
while True:
    success, img = cap.read()

    # img_out = segmentor.removeBG(img, imgBg=(255, 255, 255), threshold=0.8)
    img_out = segmentor.removeBG(img, imgBg=background_images[image_index], threshold=0.8)
    img_stacked = cvzone.stackImages([img, img_out], cols=2, scale=1)
    _, img_stacked = fps_read.update(img_stacked, color=(0, 255, 0))
    cv2.imshow("Image", img_stacked)
    key = cv2.waitKey(1)
    print(image_index)
    if key == ord('a'):
        image_index -= 1
    elif key == ord('d'):
        image_index += 1
    elif key == ord('q'):
        break
