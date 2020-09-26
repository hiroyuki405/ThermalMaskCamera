# サーモカメラ
import time
import board
import busio
import adafruit_mlx90640
import numpy as np
from PIL import Image #pip install Pillow
import smbus
import time
import cv2
from typing import List
from enum import IntEnum,auto

THERMO_RES=768
I2C_HZ=800000
TEMP_OFFSET=2
INTERVAL_UPDATE=0.3
MLX_WIDHT=32
MLX_HEIGHT=24
RES_GAIN=30 # もとのサーマルカメラの解像度から何倍まで大きくするか
COLOR_MAP_SELECT=1
TEMP_MIN = 26
TEMP_MAX = 39
COLOR_MAP_TABLE = [
    ['COLORMAP_AUTUMN',  cv2.COLORMAP_AUTUMN ],
    ['COLORMAP_JET',     cv2.COLORMAP_JET    ],
    ['COLORMAP_WINTER',  cv2.COLORMAP_WINTER ],
    ['COLORMAP_RAINBOW', cv2.COLORMAP_RAINBOW],
    ['COLORMAP_OCEAN',   cv2.COLORMAP_OCEAN  ],
    ['COLORMAP_SUMMER',  cv2.COLORMAP_SUMMER ],
    ['COLORMAP_SPRING',  cv2.COLORMAP_SPRING ],
    ['COLORMAP_COOL',    cv2.COLORMAP_COOL   ],
    ['COLORMAP_HSV',     cv2.COLORMAP_HSV    ],
    ['COLORMAP_PINK',    cv2.COLORMAP_PINK   ],
    ['COLORMAP_HOT',     cv2.COLORMAP_HOT    ]
]
class CVT_THERMO_POS_RESULT(IntEnum):
    SUCCESS=0
    OVER_LEFT=auto()
    OVER_RIGHT=auto()

def bin2temp(bin):
    '''
    サーモカメラから取得したバイナリデータを温度行列に変換
    '''
    bin = np.array(bin)
    bin = bin.reshape([MLX_HEIGHT,MLX_WIDHT])
    bin = np.fliplr(bin)
    bin = cv2.resize(bin, (MLX_WIDHT*RES_GAIN, MLX_HEIGHT*RES_GAIN))
    temp_map = Image.fromarray(np.float32(bin))
    temp_map = np.asarray(temp_map) + TEMP_OFFSET
    return temp_map

def generate_color_img(temp_map):

    '''
    温度データの行列から擬似カラー画像を生成する
    '''
    height, width = temp_map.shape
    dataclip = np.clip(temp_map, TEMP_MIN, TEMP_MAX)
    dataclip = dataclip - TEMP_MIN
    dataclip = (dataclip / (TEMP_MAX-TEMP_MIN)) * 255
    dataclip = np.round(dataclip)    
    gray_img  = np.array( dataclip, dtype="uint8")
    color_img = cv2.applyColorMap(gray_img, COLOR_MAP_TABLE[COLOR_MAP_SELECT % len(COLOR_MAP_TABLE)][1])
    return color_img

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

class MLX90640():
    '''
    MLX90640のサーマルカメラ取得クラス
    '''
    def __init__(self,eco=False):
        '''
        Parameters
        ----------
            eco:疑似カラー画像を生成しないモード
        '''
        self.eco = eco
        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=I2C_HZ)
        self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
        self.buffer = [0] * THERMO_RES
        self.last_update_tm = time.time()
        self.color_img = []
        self.temp_mat = []
        print("MLX addr detected on I2C")
        print([hex(i) for i in self.mlx.serial_number])


    def world2thermo_pos(self, pos, offset_x, offset_y):
        '''
        RGB画像の特定座標をサーモカメラの座標に変換する
        Parameters
        ----------
            pos:RGB画像中の座標
            offset_x：オフセットX座標
            offset_y：オフセットY座標
        Returns
        -------
            結果(CVT_THERMO_POS_RESULT), 変換後のx位置、変換後のy位置
        '''
        tp_w, tp_h = self.get_res()
        pos_x, pos_y = pos
        if pos_x <= offset_x :
        # if pos_x <= offset_x or pos_y <= offset_y :
            return CVT_THERMO_POS_RESULT.OVER_LEFT, None, None
        if pos_x >=( offset_x + tp_w) : # サーモカメラの解像度内に座標が存在するか
            return CVT_THERMO_POS_RESULT.OVER_RIGHT, None, None
        tp_x = pos_x - offset_x
        tp_y = pos_y - offset_y
        return CVT_THERMO_POS_RESULT.SUCCESS, tp_x, tp_y 



    def get_res(self):
        '''
        サーマルカメラの解像度取得
        Returns
        ---------
            width, height
        '''
        w = MLX_WIDHT * RES_GAIN
        h = MLX_HEIGHT * RES_GAIN
        return w, h

    def update_thermo(self):
        '''
        サーモ情報を一定間隔で更新する。
        Returns
        -------
            結果
            サーマルカメラの疑似カラー画像sh樹特
            サーマルカメラのデータをnumpy形式で取得
        '''
        played  =  time.time() - self.last_update_tm 
        if  played >= INTERVAL_UPDATE:
            self.mlx.getFrame(self.buffer) # サーモデータ取得
            self.temp_mat = bin2temp(self.buffer) # バイナリデータから温度データに変換
            if not self.eco: # エコモードが有効のときはカラー画像を生成しない
                self.color_img = generate_color_img(self.temp_mat) # 温度データから擬似カラー画像を取得
            self.last_update_tm = time.time()
            return True, self.color_img, self.temp_mat
        else:
            return False, None, None

    def get_blend(self, frame, offset_x, offset_y):
        '''
        RGB画像とサーモカメラの疑似カラー画像をブレンドする
        Parameters
        ----------
            frame:RGB画像
            offset_x:オフセットX
            offset_y：オフセットY
        Returns
        -------
            結果,　ブレンド画像
        '''
        if len(self.temp_mat) <= 0:
            return False, None
        self.color_img = generate_color_img(self.temp_mat) # 温度データから擬似カラー画像を取得
        tmp_rgb = frame.copy()
        tmp_temp =self.color_img.copy()
        tmp_temp = cv2pil(tmp_temp)
        tmp_rgb = cv2pil(tmp_rgb)
        rgb_cp = tmp_rgb.copy()            
        tmp_rgb.paste(tmp_temp, (offset_x, offset_y))
        blend = pil2cv(tmp_rgb)
        blend = cv2.addWeighted(blend, 0.5, frame, 0.5, 0)
        return True, blend
        

    def get_area_max_temp(self, left, top, right, bottom):
        '''
        指定範囲内の最大温度と疑似カラー画像を取得する。
        Parameters
        ----------
            Returns:範囲内の最大温度、トリミング疑似色画像
        '''
        left = left if left > 0 else 0
        right = right if right > 0 else 0
        top = top if top > 0 else 0
        bottom = bottom if bottom > 0 else 0
        trim_temp = self.temp_mat[left:right, top:bottom]
        if trim_temp.size <= 5:
            return 0,[]
        max_tmp = np.max(trim_temp)
        if not self.eco :
            trim_img  = self.color_img[left:right, top:bottom]
            return max_tmp, trim_img
        else :
            return max_tmp, []
       


if __name__ == "__main__":
    mlx = MLX90640()
    while True: 
        ret, color, temp = mlx.update_thermo() # MLX90640から疑似カラー画像と温度マップ取得
        if ret :
            # 指定エリアをトリミングしてエリア内の最大温度取得
            max_temp, trim_img = mlx.get_area_max_temp(20, 20, 300, 300) 

            ## 表示
            cv2.imshow("Color Temp Image", color)
            cv2.imshow("Triming Temp Image", trim_img)

            ## 画像保存
            cv2.imwrite("color_img.png", color)
            cv2.imwrite("trim_img.png", trim_img)
            print(f"Area Max Temp:{max_temp}") # 最大温度
            cv2.waitKey(1)