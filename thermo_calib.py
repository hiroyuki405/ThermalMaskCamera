## RGBカメラの映像とサーマルカメラの映像を重ねて
## キャリブレーションを行う
import thermo
import cv2
from PIL import Image
import numpy as np

CAMERA_FPS=10
CAMERA_WIDTH=1280
CAMERA_HEIGT=720
BLEND_X=int((1280-960)/2) + 40
BLEND_Y=+30
def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

from PIL import Image
import cv2

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def main():
    mlx = thermo.MLX90640()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGT)
    start = False
    count = 0
    while True:
        cv2.waitKey(1)
        ret_temp,ps_color, _ = mlx.update_thermo()
        if ret_temp:
            buf_temp_color = ps_color.copy()
        ret_frame, frame = cap.read()
        if start:
            
            tmp_rgb = frame.copy()
            tmp_temp = buf_temp_color.copy()
            tmp_temp = cv2pil(tmp_temp)
            tmp_rgb = cv2pil(tmp_rgb)
            rgb_cp = tmp_rgb.copy()            
            tmp_rgb.paste(tmp_temp, (BLEND_X, BLEND_Y))
            blend = pil2cv(tmp_rgb)
            blend = cv2.addWeighted(blend, 0.5, frame, 0.5, 0)
            x = BLEND_X
            y = BLEND_Y
            cv2.circle(blend, (x, y), 5, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
            ret, new_x, new_y = mlx.world2thermo_pos((x, y), BLEND_X, BLEND_Y)
            cv2.circle(blend, (new_x, new_y), 5, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
            cv2.imshow("blend", blend)
            # cv2.imwrite(f"blend_{count}.jpg", blend)

        if ret_temp:
            start = True
        #     cv2.imshow("temp", ps_color)
        # cv2.imshow("rgb", frame)
        count += 1



if __name__ == "__main__":
    main()