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

# PGIE_CLASS_ID_VEHICLE = 0
# PGIE_CLASS_ID_BICYCLE = 1
# PGIE_CLASS_ID_PERSON = 2
# PGIE_CLASS_ID_ROADSIGN = 3
PGIE_CLASS_ID_NO_MASK = 0
PGIE_CLASS_ID_MASK = 1
g_count = 0
pgie_classes_str= ["NoMask", "Mask"]

def draw_bounding_boxes(image,obj_meta,confidence):
    confidence='{0:.2f}'.format(confidence)
    rect_params=obj_meta.rect_params
    top=int(rect_params.top)
    left=int(rect_params.left)
    width=int(rect_params.width)
    height=int(rect_params.height)
    obj_name=pgie_classes_str[obj_meta.class_id]
    image=cv2.rectangle(image,(left,top),(left+width,top+height),(0,0,255,0),2)
    # Note that on some systems cv2.putText erroneously draws horizontal lines across the image
    image=cv2.putText(image,obj_name+',C='+str(confidence),(left-10,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255,0),2)
    return image

def osd_sink_pad_buffer_probe(pad,info,u_data):
    global g_count
    # print(f"gcount:{g_count}")
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
        while l_obj is not None:
            try:
                det = True
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
                print(f"ID:{obj_meta.object_id} class id:{obj_meta.class_id} conf:{obj_meta.confidence}")
                msg_meta=pyds.alloc_nvds_event_msg_meta()
                top =  int(obj_meta.rect_params.top)
                left = int( obj_meta.rect_params.left)
                w =  int(obj_meta.rect_params.width)
                h = int(obj_meta.rect_params.height)
                dbg = f"top:{top}, left:{left}, w:{w}, h:{h}, v:{w*h}"
                if obj_meta.class_id == PGIE_CLASS_ID_NO_MASK:
                    mark = "検温中  :{}".format(dbg)
                else:
                    mark = "マスクを外すと検温が開始されます。  :{}".format(dbg)

                n_frame=pyds.get_nvds_buf_surface(hash(gst_buffer),frame_meta.batch_id)
                frame_image=np.array(n_frame,copy=True,order='C')
                #covert the array into cv2 default color format
                frame_image=cv2.cvtColor(frame_image,cv2.COLOR_RGBA2BGRA)
                frame_image=draw_bounding_boxes(frame_image,obj_meta,obj_meta.confidence)
                # n_frame = Image.fromarray(frame_image)
                # cv2.imshow("ttt", frame_image)
                # cv2.waitKey(1)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        if det :
            display_meta.num_labels = 2
            top_title = display_meta.text_params[0]
            person_msg = display_meta.text_params[1]
            person_msg.display_text = mark
            person_msg.x_offset=left
            person_msg.y_offset=top
            person_msg.font_params.font_name= "Serif"
            person_msg.font_params.font_size= 18
            person_msg.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            person_msg.set_bg_clr = 1
            person_msg.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            person_msg.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        else :
            display_meta.num_labels = 1
            top_title = display_meta.text_params[0]

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
