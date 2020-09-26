#https://learn.adafruit.com/adafruit-mlx90640-ir-thermal-camera/python-circuitpython
import time
import board
import busio
import adafruit_mlx90640
import numpy as np
from PIL import Image #pip install Pillow
import smbus
import time
import cv2          #  測定値　実温度
OFFSET=2   # 34.2 - 36.6 #60cm
 
colormap_table = [
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
colormap_table_count = 1
def array2color(data,min,max):
    height, width = data.shape
    dataclip = np.clip(data, min, max)
    dataclip = dataclip - min
    dataclip = (dataclip / (max-min)) * 255
    dataclip = np.round(dataclip)    
    data = np.array( dataclip, dtype="uint8")
    return data
    LUT = np.zeros((256, 1, 3), dtype = "uint8") # ルックアップテーブルの配列を作成 
    dst = np.zeros((height, width, 3), dtype = "uint8") # 変換した画像を保存するための配列を作成
   
    # 青成分のルックアップテーブルを作成
    for i in range(256):
        if i < 256 / 6:
            LUT[i][0][0] = 255
        elif i < 256 * 2 / 6:
            LUT[i][0][0] = 255 - (i - 256 * 2 / 6) * 6
        elif i < 256 * 4 / 6:
            LUT[i][0][0] = 0
        elif i < 256 * 5 / 6:
            LUT[i][0][0] = (i - 256 * 4 / 6) * 6
        else:
            LUT[i][0][0] = 255
    
    # 緑成分のルックアップテーブルを作成
    for i in range(256):
        if i < 256 / 6:
            LUT[i][0][1] = i * 6
        elif i < 256 * 3 / 6:
            LUT[i][0][1] = 255
        elif i < 256 * 4 / 6:
            LUT[i][0][1] = 255 - (i - 256 * 3 / 6) * 6
        else:
            LUT[i][0][1] = 0
    
    # 赤成分のルックアップテーブルを作成
    for i in range(256):
        if i < 256 * 2 / 6:
            LUT[i][0][2] = 0
        elif i < 256 * 3 / 6:
            LUT[i][0][2] = (i - 256 * 2 / 6) * 6
        elif i < 256 * 5 / 6:
            LUT[i][0][2] = 255 - (i - 256 * 5 / 6) * 6
    
    for i in range(3): # ３チャンネル分（青・緑・赤）
        for j in range(height): # 画像の高さ分
            for k in range(width): # 画像の幅分
                dst[j][k][i] = LUT[data[j][k]][0][i] # ルックアップテーブルの値を反映

    return dst
# PRINT_TEMPERATURES = False
# PRINT_ASCIIART = True    
PRINT_TEMPERATURES = True
PRINT_ASCIIART = False

i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)

mlx = adafruit_mlx90640.MLX90640(i2c)
print("MLX addr detected on I2C")
print([hex(i) for i in mlx.serial_number])

mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ

frame = [0] * 768
cap = cv2.VideoCapture(0)
while True:
    stamp = time.monotonic()
    try:
        w=32
        h=24
        k=30
        ret, pic = cap.read()
        mlx.getFrame(frame)
        asfr = frame
        asfr = np.array(asfr)
        asfr = asfr.reshape([h,w])
        asfr = np.fliplr(asfr)
        # asfr = np.fliplr(np.flip(asfr)
        asfr = cv2.resize(asfr, (w*k, h*k))
        
        t_mat = Image.fromarray(np.float32(asfr))
        # t_mat = t_mat.resize((w*k,h*k), Image.BICUBIC)
        t_mat = np.asarray(t_mat) + OFFSET
        col = array2color(t_mat, 26, 39)
        col = cv2.applyColorMap(col, colormap_table[colormap_table_count % len(colormap_table)][1])
        # col = array2color(t_mat, 10, 40)
        # col = cv2.resize(col, (w*k, h*k))
        h = col.shape[0]
        w = col.shape[1]
        cv2.imwrite("./img.png",col)
        cv2.imwrite("./pic.png",pic)
        cv2.imshow("co", col)
        cv2.waitKey(1)
        print(f"Center:{t_mat[12][16]} Max:{np.max(t_mat)}  Min:{np.min(t_mat)} H:{h}  W:{w}")

    except ValueError:
        # these happen, no biggie - retry
        continue
    # print("Read 2 frames in %0.2f s" % (time.monotonic() - stamp))
    # for h in range(24):
    #     for w in range(32):
    #         t = frame[h * 32 + w]
    #         if PRINT_TEMPERATURES:
    #             print("%0.1f, " % t, end="")
    #         if PRINT_ASCIIART:
    #             c = "&"
    #             # pylint: disable=multiple-statements
    #             if t < 20:
    #                 c = " "
    #             elif t < 23:
    #                 c = "."
    #             elif t < 25:
    #                 c = "-"
    #             elif t < 27:
    #                 c = "*"
    #             elif t < 29:
    #                 c = "+"
    #             elif t < 31:
    #                 c = "x"
    #             elif t < 33:
    #                 c = "%"
    #             elif t < 35:
    #                 c = "#"
    #             elif t < 37:
    #                 c = "X"
    #             # pylint: enable=multiple-statements
    #             print(c, end="")
    #     print()
    # print()