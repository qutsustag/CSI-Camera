#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn, Aymeric Dujardin
@date: 20180911
"""
# pylint: disable=R, W0401, W0614, W0703
import os
import sys
import time
import datetime
import logging
import random
from random import randint
import math
import statistics
import getopt
from ctypes import *
import numpy as np
import cv2

import cv2
import threading
import numpy as np

import time
import datetime

class CSI_Camera:

    def __init__(self):
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

    def open(self, gstreamer_pipeline_string):
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            # Grab the first frame to start the video capturing
            self.grabbed, self.frame = self.video_capture.read()

        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)


    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        # Kill the thread
        self.read_thread.join()
        self.read_thread = None

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()


""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080
"""


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def run_cameras():

    h=0; m=0; s=25
    countdown(h, m, s)

#    window_title = "Dual CSI Cameras"
    left_camera = CSI_Camera()
    left_camera.open(
        gstreamer_pipeline(
            sensor_id=0,
            capture_width=3264,
            capture_height=2464,
            framerate=21,
            flip_method=2,
            display_width=3264,
            display_height=2464,
        )
    )
    left_camera.start()

    right_camera = CSI_Camera()
    right_camera.open(
        gstreamer_pipeline(
            sensor_id=1,
            capture_width=3264,
            capture_height=2464,
            framerate=21,
            flip_method=2,
            display_width=3264,
            display_height=2464,
        )
    )
    right_camera.start()

    ii = 0

    if left_camera.video_capture.isOpened() and right_camera.video_capture.isOpened():

#        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                _, left_image = left_camera.read()
                _, right_image = right_camera.read()

                # Use numpy to place images next to each other
#                camera_images = np.hstack((left_image, right_image)) 
#                camera_images = left_image
                camera_images = right_image

                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user

                timestr = time.strftime("%Y%m%d-%H%M%S")

#                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
#                    cv2.imshow(window_title, camera_images)
#                else:
#                    break

                # This also acts as
                #keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                #if keyCode == 27:
                    #break

                ii += 1

#                if cv2.waitKey(1) & 0xFF == ord('q'):
                if ii < 9:
                    save_img_path_org = "/media/liam/878A-A1C61/stereophenocam/imgraw/"
                    status = cv2.imwrite(save_img_path_org + timestr + '_left.jpg', left_image)
                    status = cv2.imwrite(save_img_path_org + timestr + '_right.jpg', right_image)
#        np.savez_compressed(save_img_path_org + timestr + '.npz', depth)

#                    np.savez_compressed(save_img_path_org + timestr + '_img_left.npz', left_image)
#                    np.savez_compressed(save_img_path_org + timestr + '_img_right.npz', right_image)

                    print("Image written to file-system: ", status)
                else:
                    break


        finally:

            left_camera.stop()
            left_camera.release()
            right_camera.stop()
            right_camera.release()
        cv2.destroyAllWindows()
    else:
        print("Error: Unable to open both cameras")
        left_camera.stop()
        left_camera.release()
        right_camera.stop()
        right_camera.release()


TOPIC = "JetsonNano/test" # Topic to which we are sending messages. You can give any value, but make sure to update that in AWS IoT policy

# Get the top-level logger object
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dict_keys = ['site', 'object', 'location', 'det_confidence']
detection_results = dict.fromkeys(dict_keys)

received_count = 0
received_all_event = threading.Event()
# is_ci = cmdUtils.get_command("is_ci", None) != None

# python3.7 aws-iot-device-sdk-python-v2/samples/pubsub.py --endpoint a2jloxkvs6jklw-ats.iot.us-west-2.amazonaws.com --ca_file root-CA.crt --cert Jetson_Nano_QUT.cert.pem
#  --key Jetson_Nano_QUT.private.key --client_id basi$d basicPubSub --topic sdk/test/Python --count 0

# Create class that acts as a countdown
def countdown(h, m, s):
 
    # Calculate the total number of seconds
    total_seconds = h * 3600 + m * 60 + s
 
    # While loop that checks if total_seconds reaches zero
    # If not zero, decrement total time by one second
    while total_seconds > 0:
 
        # Timer represents time left on countdown
        timer = datetime.timedelta(seconds = total_seconds)
        
        # Prints the time left on the timer
        print(timer, end="\r")
 
        # Delays the program one second
        time.sleep(1)
 
        # Reduces total time by one second
        total_seconds -= 1
 
    print("Bzzzt! The countdown is at zero seconds!")

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("darknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                log.info("Flag value '"+tmp+"' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # log.info(os.environ.keys())
            # log.warning("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            log.warning("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            log.warning("Environment variables indicated a CPU run, but we didn't find `" +
                        winNoGPUdll+"`. Trying a GPU run anyway.")
else:
    lib = CDLL("/home/liam/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(
    c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            name_tag = meta.names[i]
        else:
            name_tag = altNames[i]
        res.append((name_tag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    """
    Performs the detection
    """
    custom_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    custom_image = cv2.resize(custom_image, (lib.network_width(
        net), lib.network_height(net)), interpolation=cv2.INTER_LINEAR)
    im, arr = array_to_image(custom_image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(
        net, image.shape[1], image.shape[0], thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    res = []
    if debug:
        log.debug("about to range")
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    name_tag = meta.names[i]
                else:
                    name_tag = altNames[i]
                res.append((name_tag, dets[j].prob[i], (b.x, b.y, b.w, b.h), i))
    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    return res


netMain = None
metaMain = None
altNames = None

def generate_color(meta_path):
    '''
    Generate random colors for the number of classes mentioned in data file.
    Arguments:
    meta_path: Path to .data file.

    Return:
    color_array: RGB color codes for each class.
    '''
    random.seed(42)
    with open(meta_path, 'r') as f:
        content = f.readlines()
    class_num = int(content[0].split("=")[1])
    color_array = []
    for x in range(0, class_num):
        color_array.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    return color_array

# python3.7 darknet_zed_iot_v5.py -c cfg/yolov4.cfg -w yolov4.weights -m cfg/coco.data -t 0.5

def main(argv):

    thresh = 0.25
    darknet_path="/home/liam/yolo/darknet/"
    config_path = darknet_path + "cfg/yolov4.cfg"
    weight_path = "/home/liam/yolov4.weights"
    meta_path = "/home/liam/coco.data"
    svo_path = None
    zed_id = 0
    publish_count = 0
    count_img = 0

    left_camera = CSI_Camera()
    left_camera.open(
        gstreamer_pipeline(
            sensor_id=0,
            capture_width=3264,
            capture_height=2464,
            framerate=21,
            flip_method=2,
            display_width=3264,
            display_height=2464,
        )
    )
    left_camera.start()

    right_camera = CSI_Camera()
    right_camera.open(
        gstreamer_pipeline(
            sensor_id=1,
            capture_width=3264,
            capture_height=2464,
            framerate=21,
            flip_method=2,
            display_width=3264,
            display_height=2464,
        )
    )
    right_camera.start()

    # mqtt_connection = build_direct_mqtt_connection(on_connection_interrupted, on_connection_resumed)

    is_ci = False
    
    connection = False
#    try:
#        connect_future = mqtt_connection.connect()

        # Future.result() waits until a result is available
#        connect_future.result()
#        print("Connected!")
#        connection = True
#    except Exception as e:
#        logging.error('Error at %s', 'division', exc_info=e)

    detection_results['site'] = "Billabilla"
    #detection_results['object'] = 14
    ## Get the (x,y) coordinates of the object
    #detection_results['location'] = [100,110]
    # Confidence value of the detection
    #detection_results['det_confidence'] = 95

    message_count = 0
    message_topic = TOPIC
    message_string = "Hello World"

    help_str = 'darknet_zed.py -c <config> -w <weight> -m <meta> -t <threshold> -s <svo_file> -z <zed_id>'
    try:
        opts, args = getopt.getopt(
            argv, "hc:w:m:t:s:z:", ["config=", "weight=", "meta=", "threshold=", "svo_file=", "zed_id="])
    except getopt.GetoptError:
        log.exception(help_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            log.info(help_str)
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg
        elif opt in ("-w", "--weight"):
            weight_path = arg
        elif opt in ("-m", "--meta"):
            meta_path = arg
        elif opt in ("-t", "--threshold"):
            thresh = float(arg)
        elif opt in ("-s", "--svo_file"):
            svo_path = arg
        elif opt in ("-z", "--zed_id"):
            zed_id = int(arg)
   

        
#    init.depth_mode = sl.DEPTH_MODE.ULTRA  # Set the depth mode to ULTRA
#    init.coordinate_units = sl.UNIT.METER
#    init.depth_maximum_distance = 40       # Set the maximum depth perception distance to 40m
#    init.camera_resolution =  sl.RESOLUTION.HD2K

    # Import the global variables. This lets us instance Darknet once,
    # then just call performDetect() again without instancing again
    global metaMain, netMain, altNames  # pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(config_path):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(config_path)+"`")
    if not os.path.exists(weight_path):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weight_path)+"`")
    if not os.path.exists(meta_path):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(meta_path)+"`")
    if netMain is None:
        netMain = load_net_custom(config_path.encode(
            "ascii"), weight_path.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(meta_path.encode("ascii"))
    if altNames is None:
        # In thon 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(meta_path) as meta_fh:
                meta_contents = meta_fh.read()
                import re
                match = re.search("names *= *(.*)$", meta_contents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as names_fh:
                            names_list = names_fh.read().strip().split("\n")
                            altNames = [x.strip() for x in names_list]
                except TypeError:
                    pass
        except Exception:
            pass

    color_array = generate_color(meta_path)

    myobj = datetime.datetime.now()
    old_day = myobj.day
    log.info("Running...")

    key = ''
    while key != 113:  # for 'q' key
        h=0; m=0; s=10
        countdown(h, m, s)

        # new_day = myobj.day
        # if new_day != old_day:
        #     print("New day ...")
        #     old_day = new_day
        #     count_img = 0

        start_time = time.time() # start time of the loop
        timestr = time.strftime("%Y%m%d-%H%M%S")
        myobj = datetime.datetime.now()

#        try:
#            connect_future = mqtt_connection.connect()
#            # Future.result() waits until a result is available
#            connect_future.result()
#            print("Connected!")
#            connection = True
#        except Exception as e:
#            logging.error('Error at %s', 'division', exc_info=e)
#            connection = False

        ii = 0

        if left_camera.video_capture.isOpened() and right_camera.video_capture.isOpened():
            print("Cameras opened, start reading ...")
            try:
                flag = True
                while True:
                    _, image_left = left_camera.read()
                    _, image_right = right_camera.read()

                    # Use numpy to place images next to each other
    #                camera_images = np.hstack((left_image, right_image)) 
    #                camera_images = left_image
                    camera_image = image_left
                           
                    # myobj = datetime.now()
                    # print("Current hour ", myobj.hour)

                    time_minutes =  myobj.hour * 60 + myobj.minute

        #            if ((time_minutes > 690) & (time_minutes < 720)):
        #                # print("Good Morning...")
        #                save_img_path_org = "/media/jetson/1DF4-D9E4/stereophenocam/imgorg/"
        #                status = cv2.imwrite(save_img_path_org + timestr + '.jpg', image)
        #                np.savez_compressed('/media/jetson/1DF4-D9E4/stereophenocam/depth/' + timestr + '.npz', depth)
        #                print("Image written to file-system: ", status)


        #            save_img_path_org = "/media/jetson/1DF4-D9E4/stereophenocam/imgorg/"
        #            status = cv2.imwrite(save_img_path_org + timestr + '.jpg', image)
        #            print("Image written to file-system: ", status)

                    ii += 1
                   
                    if (ii < 5):
                        timestr = time.strftime("%Y%m%d-%H%M%S")

                        # print("Good Morning...")
                        save_img_path_org = "/media/liam/878A-A1C61/stereophenocam/imgorg/"
                        status = cv2.imwrite(save_img_path_org + timestr + '_right.jpg', image_left)
                        status = cv2.imwrite(save_img_path_org + timestr + '_left.jpg', image_right)
                        
                        # np.savez_compressed(save_img_path_org + timestr + '_img_right.npz', image_left)
                        # np.savez_compressed(save_img_path_org + timestr + '_img_left.npz', image_right)

                        print("Image written to file-system: ", status)
                        # depthpath_org = save_img_path_org + timestr + '.npz'
                        imgpath_org = save_img_path_org + timestr + '_left.jpg'
                        imgpath_right_org = save_img_path_org + timestr + '_right.jpg'
                        
                        
                        # with open(save_text_path + timestr + '.txt', 'a') as f:
                        #     f.write("Camera Model: " + str(cam_model)+'\n')
                        #     f.write("Serial Number: " + str(info.serial_number)+'\n')
                        #     f.write("Camera Firmware: " + str(info.camera_configuration.firmware_version)+'\n')
                        #     f.write("Sensors Firmware: " + str(info.sensors_configuration.firmware_version)+'\n')

                        #     f.write("Quaternion: " + str(quaternion[0])+' '+str(quaternion[1])+' '+str(quaternion[2])+' '+str(quaternion[3])+'\n')
                        #     f.write("Rotation xyz (degree): " + str(zed_roll_x_deg)+' '+str(zed_pitch_y_deg)+' '+str(zed_yaw_z_deg)+'\n')
                        #     f.write("Magetic: " + str(magnetic_field_calibrated[0])+' '+str(magnetic_field_calibrated[1])+' '+str(magnetic_field_calibrated[2])+'\n')
                        #     f.write("Heading: " + str(sensors_data.get_magnetometer_data().magnetic_heading)+' '+str(sensors_data.get_magnetometer_data().magnetic_heading_accuracy)+'\n')
                        #     f.write("Pressure: " + str(pressure_data )+'\n')

                        # message_json = json.dumps(detection_results)
                        connection = False
                        

                        # Do the detection
                        detections = detect(netMain, metaMain, image_left, thresh)
                        isperson = False

                        detection_label = ["person", "cow"]

                        log.info(chr(27) + "[2J"+"**** " + str(len(detections)) + " Results ****")
                        for detection in detections:
                            label = detection[0]
            #                if (label != "person"):
                            if (label not in detection_label):
                                continue
            #                isperson = True
                            confidence = detection[1]
                            if (confidence < 0.6):
                                continue
                            isperson = True
                            pstring = label+": "+str(np.rint(100 * confidence))+"%"
                            log.info(pstring)
                            bounds = detection[2]
                            y_extent = int(bounds[3])
                            x_extent = int(bounds[2])
                            # Coordinates are around the center
                            x_coord = int(bounds[0] - bounds[2]/2)
                            y_coord = int(bounds[1] - bounds[3]/2)

            #                obj_width = depth[y_coord,x_coord,0]  - depth[y_coord,x_coord + x_extent,0]
            #                obj_height = depth[y_coord+y_extent,x_coord,1]  - depth[y_coord,x_coord,1]

            #                point_coord  = point_cloud_mat.get_value(x_coord, y_coord)
            #                point_width  = point_cloud_mat.get_value(x_coord+x_extent, y_coord)
            #                point_height  = point_cloud_mat.get_value(x_coord, y_coord+y_extent)


            #                obj_width = point_coord[0] - point_width[0]
            #                obj_height = point_coord[1] - point_width[1]

            #                print("Object width, height (m):", obj_width, obj_height)

            #                print(x_coord,y_coord)
            #                print(x_coord+x_extent,y_coord)
            #                print(depth.shape)
            #                print(depth[x_coord,y_coord,0])
            #                print(depth[x_coord+x_extent,y_coord,0])

                            #boundingBox = [[x_coord, y_coord], [x_coord, y_coord + y_extent], [x_coord + x_extent, y_coord + y_extent], [x_coord + x_extent, y_coord]]
                            thickness = 1
                            # x, y, z = get_object_depth(depth, bounds)
            #                print(x,y,z)
                            # distance = math.sqrt(x * x + y * y + z * z)
                            # distance = "{:.2f}".format(distance)
            #                cv2.rectangle(image, (x_coord - thickness, y_coord - thickness),
            #                              (x_coord + x_extent + thickness, y_coord + (18 + thickness*4)),
            #                              color_array[detection[3]], -1)
            #                cv2.putText(image, label + " " +  (str(distance) + " m"),
            #                            (x_coord + (thickness * 4), y_coord + (10 + thickness * 4)),
            #                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #                cv2.rectangle(image, (x_coord - thickness, y_coord - thickness),
            #                              (x_coord + x_extent + thickness, y_coord + y_extent + thickness),
            #                              color_array[detection[3]], int(thickness*2))

                            message = "{} [{}]".format(message_string, publish_count)
                            print("Publishing message to topic '{}': {}".format(message_topic, message))
                            # message_json = json.dumps(message)
                            detection_results['object'] = label
                            detection_results['location'] = [x_coord,y_coord,x_extent,y_extent]
                            detection_results['det_confidence'] = confidence
                            # line = (label, x_coord, y_coord, x_extent, y_extent, confidence)
            #                timestr = time.strftime("%Y%m%d-%H%M%S")
                            save_text_path = "/media/liam/878A-A1C61/stereophenocam/detection/"

                            with open(save_text_path + timestr + '.txt', 'a') as f:
                                f.write(label+' '+str(x_coord)+' '+str(y_coord)+' '+str(x_extent)+' '+str(y_extent)+' '+str(confidence)+'\n')
            
                            # message_json = json.dumps(detection_results)
            #                mqtt_connection.publish(
            #                    topic=message_topic,
            #                    payload=message_json,
            #                    qos=mqtt.QoS.AT_LEAST_ONCE)
                            # time.sleep(1)
                            publish_count += 1

            #             if isperson:
            # #                cv2.imshow("ZED", image)
            #                 save_img_path = "/media/liam/STORE N GO/stereophenocam/img/"
            #                 # status = cv2.imwrite(save_img_path + timestr + '.jpg', image)
            #                 status = cv2.imwrite(save_img_path + 'detections.jpg', image_left)
            #                 status = cv2.imwrite(save_img_path + 'detections_right.jpg', image_right)
            #                 print("Image written to file-system: ", status)

            #                 # np.savez_compressed('/media/jetson/1DF4-D9E4/stereophenocam/depth/depth.npz', depth)
            #                 time.sleep(1)

            # #                key = cv2.waitKey(5)
            #                 log.info("FPS: {}".format(1.0 / (time.time() - start_time)))
            #                 time.sleep(1)
            #                 # s3 = boto3.client('s3')
            #                 textpath = save_text_path + timestr + '.txt'
            #                 print(textpath)

                            # myobj = datetime.now()
                            # print("Current hour ", myobj.hour)

            #                minutes = myobj.hour * 60 + myobj.minute

            #                if myobj.hour == 12:
            #                    if myobj.minute < 30:
            #                        # print("Good Morning...")
            #                        status = cv2.imwrite(save_img_path + timestr + '.jpg', image)
            #                        print("Image written to file-system: ", status)


            #                if ((minutes >= 690) & (minutes <= 750)): # 11.30 to 12.30
            #                    # print("Good Morning...")
            #                    status = cv2.imwrite(save_img_path + timestr + '.jpg', image)
            #                    print("Image written to file-system: ", status)


                            # if count_img < 10:
                            #     # print("Good Morning...")
                            #     count_img += 1
                            #     status = cv2.imwrite(save_img_path + timestr + '_left.jpg', image_left)
                            #     status = cv2.imwrite(save_img_path + timestr + '_right.jpg', image_right)
                            #     print("Image written to file-system: ", status)
                            #     # np.savez_compressed('/media/jetson/1DF4-D9E4/stereophenocam/depth/' + timestr + '.npz', depth)
                            #     time.sleep(1)


                            # # imgpath = save_img_path + timestr + '.jpg'
                            # imgpath = save_img_path + 'detections.jpg'
                            # imgpath_right = save_img_path + 'detections_right.jpg'
                            # print(imgpath)
                            # depthpath = "/media/jetson/1DF4-D9E4/stereophenocam/depth/depth.npz"
            #                connection = boto3.connect()
                            
            
                    else:
                        # left_camera.stop()
                        #left_camera.release()
                        # right_camera.stop()
                        # right_camera.release()
                        # exit(1)
                        break
                        #flag = False
            finally:
                # print("Error: Could not read cameras...")
                left_camera.stop()
                left_camera.release()
                right_camera.stop()
                right_camera.release()
                # key = cv2.waitKey(5)
                exit(1) 
                
            # key = cv2.waitKey(5)  
            
        else:
            print("Error: Unable to open both cameras")
            left_camera.stop()
            left_camera.release()
            right_camera.stop()
            right_camera.release()
            # key = cv2.waitKey(5)
            exit(1)

        # key = cv2.waitKey(5) 
        break

    cv2.destroyAllWindows()

    # print("Error: Unable to open both cameras")
    
    log.info("\nFINISH")

    # Disconnect
    print("Disconnecting...")
    # disconnect_future = mqtt_connection.disconnect()
    # disconnect_future.result()
    print("Disconnected!")


if __name__ == "__main__":
    main(sys.argv[1:])
