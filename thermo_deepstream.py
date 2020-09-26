################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

#!/usr/bin/env python
import datetime
import shutil
import os
import cv2
import traceback
import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from common.is_aarch_64 import is_aarch64
from common.utils import long_to_int
from common.bus_call import bus_call
import pyds
import numpy as np
from PIL import Image
import thermo
from enum import IntEnum,auto
class THERMO_RESULT(IntEnum):
    SUCCESS=0#取得成功
    AREA_OVER=auto() #エリア外
    SMALL_AREA=auto() # 面積が小さい＝遠すぎ
    OVER_AREA=auto() # 面積が大きい＝近すぎ
    LEFT_OVER=auto() # 左にはみ出すぎ
    RIGHT_OVER=auto() #右にはみ出すぎ

BLEND_X=int((1280-960)/2) + 40
BLEND_Y=+30

# PGIE_CLASS_ID_VEHICLE = 0
# PGIE_CLASS_ID_BICYCLE = 1
# PGIE_CLASS_ID_PERSON = 2
# PGIE_CLASS_ID_ROADSIGN = 3
PGIE_CLASS_ID_NO_MASK = 0
PGIE_CLASS_ID_MASK = 1
g_count = 0
pgie_classes_str= ["NoMask", "Mask"]
PIXEL_W=1280
PIXEL_H=720
mlx:thermo.MLX90640 = thermo.MLX90640(eco=True)
OFFSET=40
MAX_THERMO_AREA= 1280*720
MIN_THERMO_AREA=0
CENTER_DIST=300
FPS=10
FOURCC = cv2.VideoWriter_fourcc(*'XVID')
dtnow = datetime.datetime.now()
filename = dtnow.strftime('log/%Y_%m_%d_%H_%M_%S.avi')
writer = cv2.VideoWriter(filename, FOURCC, FPS, (840, 640))
MAX_FRAME=5
frame_count = 0

def gen_new_video():
    global frame_count, writer
    print("Generaret Video!")
    dtnow = datetime.datetime.now()
    filename = dtnow.strftime('log/%Y_%m_%d_%H_%M_%S.avi')
    frame_count = 0
    writer.release()
    writer = cv2.VideoWriter(filename, FOURCC, FPS, (840, 640))

def draw_bounding_boxes(image,obj_meta,confidence,top,left,w, h, msg,msg2):
    confidence='{0:.2f}'.format(confidence)
    rect_params=obj_meta.rect_params
    # top=int(rect_params.top)
    # left=int(rect_params.left)
    # width=int(rect_params.width)
    # height=int(rect_params.height)
    top=top
    left=left
    width=w
    height=h
    obj_name=pgie_classes_str[obj_meta.class_id]
    image=cv2.rectangle(image,(left,top),(left+width,top+height),(45,255,0,0),2)
    # Note that on some systems cv2.putText erroneously draws horizontal lines across the image
    image=cv2.putText(image, msg2,(10,50),         cv2.FONT_HERSHEY_SIMPLEX,1,(45,255,0,0),2)
    image=cv2.putText(image, msg, (left-10,top-10),cv2.FONT_HERSHEY_SIMPLEX,1,(45,255,0,0),2)
    return image

def getOffset(v,dist_2d, xright):
    '''
    面積からオフセット値を取得
    '''
    offset_v = (0.2/50000 * v)*0.8 - 0.1 - 0.1 - 0.76
    offset_2d = 0.0021985*dist_2d
    offset = -(offset_v - offset_2d )
    alpha = 0
    offset = offset - 0.7
    offset_right = 0
    if xright:
        offset_right = (0.018261 * dist_2d)*1.2
        offset += offset_right
    if offset_v <= -0.45:
        alpha = ((1.3895 * -(offset_v)))
        offset += alpha
    print("v:{:.5f} 2doffset:{:.5f} dist_2d:{} r_off:{} alpha:{}".format(offset_v, offset_2d, dist_2d,offset_right, alpha) )
    return offset

def get_temp(left, top , w, h, mlx:thermo.MLX90640):
    '''
    温度を取得する
    Parameters
    ----------
        RGB画像のX座標,　Y座標、幅、高さ、MLXオブジェクト
    Returns
    -------
        結果（THERMO_RESULT）、サーモ面積、温度
    '''
    ret_s, t_left, t_top =mlx.world2thermo_pos((left,top), BLEND_X, BLEND_Y)
    ret_e, t_right, t_bottom =mlx.world2thermo_pos((left+w,top+h), BLEND_X, BLEND_Y)
    if ret_s == thermo.CVT_THERMO_POS_RESULT.OVER_LEFT:
        return THERMO_RESULT.LEFT_OVER, None, None
    if ret_e == thermo.CVT_THERMO_POS_RESULT.OVER_RIGHT :
        return THERMO_RESULT.RIGHT_OVER, None, None

    area_v = (t_right - t_left) * (t_bottom - t_top)
    print(f"area_v:{area_v}")
    if area_v <= MIN_THERMO_AREA:
        return THERMO_RESULT.SMALL_AREA, area_v, None
    elif area_v >= MAX_THERMO_AREA:
        return THERMO_RESULT.OVER_AREA, area_v, None
    
    temp, _ = mlx.get_area_max_temp(t_left, t_top, t_right, t_bottom)
    return THERMO_RESULT.SUCCESS, area_v, temp

def osd_clip(left, top , w, h):
    left = left if left >= 0 else 0
    left = left if left <= PIXEL_W else PIXEL_W
    top = top if top >= 0 else 0
    top = top if top <= PIXEL_H else PIXEL_H
    w = w if (left+w) <= PIXEL_W else (PIXEL_W-w)
    w = w if w >= 0 else 0
    h = h if (top+h) <= PIXEL_H else (PIXEL_H - w)
    h = h if h >= 0 else 0
    return left, top, w, h

def osd_sink_pad_buffer_probe(pad,info,u_data):
    global g_count,writer, frame_count
    # global mlx
    
    # if ret:
    #     cv2.imwrite("colo.png", color_mat)

    # print(f"gcount:{g_count}")
    # if frame_count >= MAX_FRAME: # 動画を分岐
    #     gen_new_video()
    frame_number=0
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_NO_MASK:0,
        PGIE_CLASS_ID_MASK:0
        # PGIE_CLASS_ID_VEHICLE:0,
        # PGIE_CLASS_ID_PERSON:0,
        # PGIE_CLASS_ID_BICYCLE:0,
        # PGIE_CLASS_ID_ROADSIGN:0
    }
    num_rects=0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable import pydsto get GstBuffer ")
        return


    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    # 画像表示
    # frame_rgba = pyds.get_nvds_buf_surface(gst_buffer, g_count)
    # cv2.imshow("Streamk",frame_rgba)
    log_count = 0

    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.glist_get_nvds_frame_meta()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            
            frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
        except StopIteration:
            break


        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        det = False
        disp_list=[]
        n_frame=pyds.get_nvds_buf_surface(hash(gst_buffer),frame_meta.batch_id)
        frame_image=np.array(n_frame,copy=True,order='C')
        frame_image=cv2.cvtColor(frame_image,cv2.COLOR_RGBA2BGRA)
        while l_obj is not None:
            try:
                ret, color_mat, temp_mat =mlx.update_thermo()
                det = True
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
                # print(f"ID:{obj_meta.object_id} class id:{obj_meta.class_id} conf:{obj_meta.confidence}")
                msg_meta=pyds.alloc_nvds_event_msg_meta()
                top =  int(obj_meta.rect_params.top)
                top_buf = top
                left = int( obj_meta.rect_params.left) - OFFSET
                left_buf = left
                w_buf = int(obj_meta.rect_params.width)
                h_buf = int(obj_meta.rect_params.height)
                w =  int(obj_meta.rect_params.width) + OFFSET * 2
                h = int(obj_meta.rect_params.height)+OFFSET
                left, top ,w ,h = osd_clip(left, top, w, h)
                center_x = int(left_buf + (int(obj_meta.rect_params.width) / 2))
                ret_therm, area_v, temp = get_temp(left, top, w, h , mlx)

                #中央からどれだけ離れているか
                center_pos = np.array([PIXEL_W/2 , PIXEL_H/2])
                face_x = left_buf + (w_buf/2)
                face_y = top_buf + (h_buf/2)
                face_pos = np.array([face_x, face_y])
                distance_c = center_pos - face_pos
                distance_c = np.linalg.norm(distance_c)
                ## 分岐処理
                if obj_meta.class_id == PGIE_CLASS_ID_MASK:
                    mark = "マスクを外すと検温が開始されます。"
                elif ret_therm == THERMO_RESULT.SUCCESS and distance_c <=CENTER_DIST:
                    offset = getOffset(area_v, distance_c, PIXEL_W/2  <= face_x) 
                    temp = temp + offset
                    center =(PIXEL_W/2 ,PIXEL_H/2)
                    # mark = "v:{} 温度{:.2f} offset:{:.5f} dist:{:.4f} c:{}".format(area_v, temp,offset, distance_c, PIXEL_W - center_x) デバッグ用
                    mark = "静止してください。 温度{:.2f}〜{:.2f}".format(temp-2,temp+2)
                elif ret_therm == THERMO_RESULT.OVER_AREA:
                    mark = "近づきすぎです。"
                elif ret_therm == THERMO_RESULT.SMALL_AREA:
                    mark = "もう少し近づいてください。"
                elif ret_therm == THERMO_RESULT.LEFT_OVER or distance_c >= CENTER_DIST:
                    mark = "中央に寄ってください。"
                elif ret_therm == THERMO_RESULT.RIGHT_OVER >= CENTER_DIST:
                    mark = "中央に寄ってください"

                #covert the array into cv2 default color format
                # print("ffs")
                # frame_image=cv2.cvtColor(frame_image,cv2.COLOR_RGBA2BGRA)
                dtnow = datetime.datetime.now()
                ts = dtnow.strftime('%Y_%m_%d  %H:%M:%S')
                mark2 = f"{ts} f:{frame_count}"
                if frame_count >= MAX_FRAME:
                    frame_count = 0
                    dtnow = datetime.datetime.now()
                    filename = dtnow.strftime('%Y_%m_%d_%H_%M_%S.jpg')
                    #frame_image=draw_bounding_boxes(frame_image,obj_meta,obj_meta.confidence, top, left, w, h, "{:.2f}".format(temp), mark2)
                    frame_r = cv2.resize(frame_image, (1280,720))
                    if log_count >= 20:
                        log_count=0 
                        cv2.imwrite(f"log/{filename}",frame_r)
                    log_count += 1

                    cv2.imwrite("te.jpg", frame_r)
                    shutil.move("te.jpg", f"mjpeg/stream.jpg")

                    print(f"save  mjpeg/{filename}")

                # writer.write(frame_r)
                frame_count += 1
                # n_frame = Image.fromarray(frame_image)
                if left_buf > 0 and top_buf > 0:
                    disp_list.append((mark, left_buf, top_buf))
                # cv2.imshow("ttt", frame_image)
                # cv2.waitKey(1)
                # print("end")
                # cv2.imwrite("tmp.jpg", frame_image)
                # cv2.imwrite("mjpeg/stream.jpg", frame_image)
                # shutil.move("tmp.jpg", "mjpeg/stream.jpg")
                # shutil.move("te.jpg", "image.jpg")


            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        
        # blend_ret, blend_img = mlx.get_blend(frame_image, BLEND_X, BLEND_Y )
        # if blend_ret:
        #     cv2.imwrite("tmp.jpg", blend_img)
        #     shutil.move("tmp.jpg", "mjpeg/stream.jpg")

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        
        display_meta.num_labels = len(disp_list) + 1
        top_title = display_meta.text_params[0]
        for index, meta in enumerate(disp_list):
            msg, left, top = meta
            person_msg = display_meta.text_params[index+1]
            person_msg.display_text = msg
            person_msg.x_offset=left
            person_msg.y_offset=top
            person_msg.font_params.font_name= "Serif"
            person_msg.font_params.font_size= 18
            person_msg.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            person_msg.set_bg_clr = 1
            person_msg.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            person_msg.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

        # print(f"display_meta.text_params size:{len(display_meta.text_params)}")
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_t
        # ext field here will return the C address of the
        # allo
        # cated string. Use pyds.get_string() to get the string content.
        #top_title.display_text = "Frahme Number={} Number of Objects={} NoMask_count={} Mask_count={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_NO_MASK], obj_counter[PGIE_CLASS_ID_MASK])
        top_title.display_text = "体温チェッカー（試作品）"


        # Now set the offsets where the string should appear
        top_title.x_offset = 10
        top_title.y_offset = 12

        # Font , font-color and font-size
        top_title.font_params.font_name = "Serif"
        top_title.font_params.font_size = 30
        # set(red, green, blue, alpha); set to White
        top_title.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        top_title.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        top_title.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        # print(pyds.get_string(top_title.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
        g_count += 1
        
    return Gst.PadProbeReturn.OK	


def main(args):
    # Check input arguments
    if len(args) != 2:
        device_cam  = "/dev/video0"
        print("Default camera Set! {}".format(device_cam))
        # sys.stderr.write("usage: %s <v4l2-device-path>\n" % args[0])
        # sys.exit(1)
    else:
        device_cam = args[1]
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)


    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element for reading from the file
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
    if not caps_v4l2src:
        sys.stderr.write(" Unable to create v4l2src capsfilter \n")

    print("Creating Video Converter \n")

    # Adding videoconvert -> nvvideoconvert as not all
    # raw formats are supported by nvvideoconvert;
    # Say YUYV is unsupported - which is the common
    # raw format for many logi usb cams
    # In case we have a camera with raw format supported in
    # nvvideoconvert, GStreamer plugins' capability negotiation
    # shall be intelligent enough to reduce compute by
    # videoconvert doing passthrough (TODO we need to confirm this)

# videoconvert to make sure a superset of raw formats are supported
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    if not vidconvsrc:
        sys.stderr.write(" Unable to create videoconvert \n")

    # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create Nvvideoconvert \n")

    caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    if not caps_vidconvsrc:
        sys.stderr.write(" Unable to create capsfilter \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # Use nvinfer to run inferencing on camera's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # Finally render the osd output
    if is_aarch64():
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

    print("Creating EGLSink \n")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    print("Playing cam %s " %device_cam )
    caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw,width=1280,height=720,framerate=10/1"))
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
    source.set_property('device', device_cam )
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('gpu-id', 0)
    # pgie.set_property('config-file-path', "yolov3tiny.txt")
    pgie.set_property('config-file-path', "mask_config.txt")
    # pgie.set_property('interval', 1)
    # pgie.set_property('model-engine-file', "deepstream/model_b1_fp16_mask.engine")

    # Set sync = false to avoid late frame drops at the display-sink
    sink.set_property('sync', False)

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(caps_v4l2src)
    pipeline.add(vidconvsrc)
    pipeline.add(nvvidconvsrc)
    pipeline.add(caps_vidconvsrc)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)
    if is_aarch64():
        pipeline.add(transform)
    # we link the elements together
    # v4l2src -> nvvideoconvert -> mux -> 
    # nvinfer -> nvvideoconvert -> nvosd -> video-renderer
    print("Linking elements in the Pipeline \n")
    source.link(caps_v4l2src)
    caps_v4l2src.link(vidconvsrc)
    vidconvsrc.link(nvvidconvsrc)
    nvvidconvsrc.link(caps_vidconvsrc)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = caps_vidconvsrc.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    if is_aarch64():
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    print("EXEPT")
    # cleanup
    pipeline.set_state(Gst.State.NULL)
if __name__ == '__main__':
    sys.exit(main(sys.argv))
