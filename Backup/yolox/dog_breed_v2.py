#!/usr/bin/env python3

"""
The code is the same as for Tiny Yolo V3 and V4, the only difference is the blob file
- Tiny YOLOv3: https://github.com/david8862/keras-YOLOv3-model-set
- Tiny YOLOv4: https://github.com/TNTWEN/OpenVINO-YOLOV4
"""

from pathlib import Path
from shutil import which
import sys
import cv2
import depthai as dai
import numpy as np
import time
from keras.applications.densenet import preprocess_input
import matplotlib.pyplot as plt


import depthai as dai
from typing import Tuple, Union, Optional, List
from string import Template
import os
from pathlib import Path

class MultiStageNN():
    script: dai.node.Script
    manip: dai.node.ImageManip
    out: dai.Node.Output # Cropped imgFrame output
    i: int = 0

    _size: Tuple[int, int]
    def __init__(self,
        pipeline: dai.Pipeline,
        detector: Union[
            dai.node.MobileNetDetectionNetwork,
            dai.node.MobileNetSpatialDetectionNetwork,
            dai.node.YoloDetectionNetwork,
            dai.node.YoloSpatialDetectionNetwork], # Object detector
        highResFrames: dai.Node.Output,
        size: Tuple[int, int],
        debug = False
        ) -> None:
        """
        Args:
            detections (dai.Node.Output): Object detection output
            highResFrames (dai.Node.Output): Output that will provide high resolution frames
        """

        self.script = pipeline.create(dai.node.Script)
        self.script.setProcessor(dai.ProcessorType.LEON_CSS) # More stable
        self._size = size

        detector.out.link(self.script.inputs['detections'])
        highResFrames.link(self.script.inputs['frames'])

        self.configMultiStageNn(debug = debug,)

        self.manip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(size)
        self.manip.setWaitForConfigInput(True)
        self.manip.setMaxOutputFrameSize(size[0] * size[1] * 3)
        self.manip.setNumFramesPool(20)
        self.script.outputs['manip_cfg'].link(self.manip.inputConfig)
        self.script.outputs['manip_img'].link(self.manip.inputImage)
        self.out = self.manip.out

    def configMultiStageNn(self,
        debug = False,
        labels: Optional[List[int]] = None,
        scaleBb: Optional[Tuple[int, int]] = None,
        ) -> None:
        """
        Args:
            debug (bool, default False): Debug script node
            labels (List[int], optional): Crop & run inference only on objects with these labels
            scaleBb (Tuple[int, int], optional): Scale detection bounding boxes (x, y) before cropping the frame. In %.
        """

        with open(Path(os.path.dirname(__file__)) / 'template_multi_stage_script.py', 'r') as file:
            code = Template(file.read()).substitute(
                DEBUG = '' if debug else '#',
                CHECK_LABELS = f"if det.label not in {str(labels)}: continue" if labels else "",
                WIDTH = str(self._size[0]),
                HEIGHT = str(self._size[1]),
                SCALE_BB_XMIN = f"-{scaleBb[0]/100}" if scaleBb else "", # % to float value
                SCALE_BB_YMIN = f"-{scaleBb[1]/100}" if scaleBb else "",
                SCALE_BB_XMAX = f"+{scaleBb[0]/100}" if scaleBb else "",
                SCALE_BB_YMAX = f"+{scaleBb[1]/100}" if scaleBb else "",
            )
            self.script.setScript(code)
            # print(f"\n------------{code}\n---------------")


# other = MultiStageNN(
#     pipeline=pipeline,
#     detector=detectionNetwork,
#     highResFrames=,
#     size=(62,62),
#     debug=True
#     )

def preproc(image, input_size, mean, std, swap=(2, 0, 1)):

    if image.getWidth() == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float16)
    return padded_img, r

def correct_bb(bb):
    if bb.xmin < 0: bb.xmin = 0.001
    if bb.ymin < 0: bb.ymin = 0.001
    if bb.xmax > 1: bb.xmax = 0.999
    if bb.ymax > 1: bb.ymax = 0.999
    return bb
def amplify_bb(bb):
    bb.xmin -= 0.05
    bb.ymin -= 0.05
    bb.xmax += 0.05
    bb.ymax += 0.05
    return bb



# Get argument first
nnPath = str((Path(__file__).parent / Path('D:/Python/Luxonis/depthai-experiments/gen2-yolo/yolox/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
if 1 < len(sys.argv):
    arg = sys.argv[1]
    if arg == "yolo3":
        nnPath = str((Path(__file__).parent / Path('../models/yolo-v3-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
    elif arg == "yolo4":
        nnPath = str((Path(__file__).parent / Path('../models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
    else:
        nnPath = arg
else:
    print("Using Tiny YoloV4 model. If you wish to use Tiny YOLOv3, call 'tiny_yolo.py yolo3'")

if not Path(nnPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# tiny yolo v4 label texts
labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]

breedMap = [
"n02091635-otterhound",
"n02102318-cocker_spaniel",
"n02101388-Brittany_spaniel",
"n02088094-Afghan_hound",
"n02085936-Maltese_dog",
"n02104365-schipperke",
"n02100877-Irish_setter",
"n02086079-Pekinese",
"n02099601-golden_retriever",
"n02100583-vizsla",
"n02102177-Welsh_springer_spaniel",
"n02093256-Staffordshire_bullterrier",
"n02106166-Border_collie",
"n02093991-Irish_terrier",
"n02109961-Eskimo_dog",
"n02110958-pug",
"n02105412-kelpie",
"n02094433-Yorkshire_terrier",
"n02097474-Tibetan_terrier",
"n02089867-Walker_hound",
"n02110627-affenpinscher",
"n02113186-Cardigan",
"n02102040-English_springer",
"n02089973-English_foxhound",
"n02098286-West_Highland_white_terrier",
"n02095570-Lakeland_terrier",
"n02087394-Rhodesian_ridgeback",
"n02101006-Gordon_setter",
"n02098413-Lhasa",
"n02099429-curly-coated_retriever",
"n02088364-beagle",
"n02108551-Tibetan_mastiff",
"n02102480-Sussex_spaniel",
"n02109525-Saint_Bernard",
"n02087046-toy_terrier",
"n02113799-standard_poodle",
"n02107683-Bernese_mountain_dog",
"n02112018-Pomeranian",
"n02091244-Ibizan_hound",
"n02090379-redbone",
"n02113624-toy_poodle",
"n02088238-basset",
"n02092002-Scottish_deerhound",
"n02107312-miniature_pinscher",
"n02110806-basenji",
"n02093754-Border_terrier",
"n02093647-Bedlington_terrier",
"n02093859-Kerry_blue_terrier",
"n02092339-Weimaraner",
"n02100735-English_setter",
"n02088632-bluetick",
"n02096585-Boston_bull",
"n02091032-Italian_greyhound",
"n02096437-Dandie_Dinmont",
"n02096051-Airedale",
"n02102973-Irish_water_spaniel",
"n02094114-Norfolk_terrier",
"n02095314-wire-haired_fox_terrier",
"n02108915-French_bulldog",
"n02098105-soft-coated_wheaten_terrier",
"n02105505-komondor",
"n02116738-African_hunting_dog",
"n02110185-Siberian_husky",
"n02111277-Newfoundland",
"n02106382-Bouvier_des_Flandres",
"n02091831-Saluki",
"n02105855-Shetland_sheepdog",
"n02106030-collie",
"n02106550-Rottweiler",
"n02097658-silky_terrier",
"n02091467-Norwegian_elkhound",
"n02085620-Chihuahua",
"n02111129-Leonberg",
"n02094258-Norwich_terrier",
"n02096177-cairn",
"n02108089-boxer",
"n02090622-borzoi",
"n02115913-dhole",
"n02111889-Samoyed",
"n02106662-German_shepherd",
"n02099712-Labrador_retriever",
"n02086646-Blenheim_spaniel",
"n02105056-groenendael",
"n02107142-Doberman",
"n02109047-Great_Dane",
"n02099267-flat-coated_retriever",
"n02107908-Appenzeller",
"n02086240-Shih-Tzu",
"n02085782-Japanese_spaniel",
"n02107574-Greater_Swiss_Mountain_dog",
"n02089078-black-and-tan_coonhound",
"n02115641-dingo",
"n02111500-Great_Pyrenees",
"n02091134-whippet",
"n02112350-keeshond",
"n02105162-malinois",
"n02093428-American_Staffordshire_terrier",
"n02113978-Mexican_hairless",
"n02097130-giant_schnauzer",
"n02112706-Brabancon_griffon",
"n02104029-kuvasz",
"n02113712-miniature_poodle",
"n02090721-Irish_wolfhound",
"n02105251-briard",
"n02101556-clumber",
"n02097209-standard_schnauzer",
"n02108422-bull_mastiff",
"n02110063-malamute",
"n02095889-Sealyham_terrier",
"n02108000-EntleBucher",
"n02112137-chow",
"n02086910-papillon",
"n02113023-Pembroke",
"n02100236-German_short-haired_pointer",
"n02105641-Old_English_sheepdog",
"n02099849-Chesapeake_Bay_retriever",
"n02097298-Scotch_terrier",
"n02096294-Australian_terrier",
"n02097047-miniature_schnauzer",
"n02088466-bloodhound",
]
DOG_INDEX = 16
syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(416, 416)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
detectionNetwork.setConfidenceThreshold(0.8)
detectionNetwork.setNumClasses(80)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
detectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
detectionNetwork.setIouThreshold(0.2)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)
camRgb.preview.link(detectionNetwork.input)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
if syncNN:
    detectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

nnOut = pipeline.create(dai.node.XLinkOut)
nnOut.setStreamName("nn")
detectionNetwork.out.link(nnOut.input)

# Script node will take the output from the face detection NN as an input and set ImageManipConfig KEEP
# to the 'recognition_manip' to crop the initial frame
# image_manip_script = pipeline.create(dai.node.Script)
# image_manip_script.setProcessor(dai.ProcessorType.LEON_CSS)
# detectionNetwork.out.link(image_manip_script.inputs['obj_in']) #Receive object

# # Remove in 2.18 and use `imgFrame.getSequenceNum()` in Script node
# detectionNetwork.passthrough.link(image_manip_script.inputs['passthrough']) #bb

# camRgb.preview.link(image_manip_script.inputs['preview'])#Receive image

# image_manip_script.setScript("""#TODO preprocessing, change color order
# import time
# msgs = dict()
# cntr = 0

# def add_msg(msg, name, seq = None):
#     global msgs
#     if seq is None:
#         seq = msg.getSequenceNum()
#     seq = str(seq)
#     # node.warn(f"New msg {name}, seq {seq}")

#     # Each seq number has it's own dict of msgs
#     if seq not in msgs:
#         msgs[seq] = dict()
#     msgs[seq][name] = msg

#     # To avoid freezing (not necessary for this ObjDet model)
#     if 15 < len(msgs):
#         node.warn(f"Removing first element! len {len(msgs)}")
#         msgs.popitem() # Remove first element

# def get_msgs():
#     global msgs
#     seq_remove = [] # Arr of sequence numbers to get deleted
#     for seq, syncMsgs in msgs.items():
#         seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair
#         # node.warn(f"Checking sync {seq}")

#         # Check if we have both detections and color frame with this sequence number
#         if len(syncMsgs) == 2: # 1 frame, 1 detection
#             for rm in seq_remove:
#                 del msgs[rm]
#             # node.warn(f"synced {seq}. Removed older sync values. len {len(msgs)}")
#             return syncMsgs # Returned synced msgs
#     return None

# def correct_bb(bb):
#     if bb.xmin < 0: bb.xmin = 0.001
#     if bb.ymin < 0: bb.ymin = 0.001
#     if bb.xmax > 1: bb.xmax = 0.999
#     if bb.ymax > 1: bb.ymax = 0.999
#     return bb

# while True:
#     time.sleep(0.001) # Avoid lazy looping
#     preview = node.io['preview'].tryGet() #Get image from camera
#     if preview is not None:
#         add_msg(preview, 'preview')

#     obj_classification = node.io['obj_in'].tryGet()
#     # node.warn(obj_classification[0].label)
#     if obj_classification is not None:
#         # TODO: in 2.18.0.0 use face_dets.getSequenceNum()
#         passthrough = node.io['passthrough'].get()
#         seq = passthrough.getSequenceNum()
#         cntr -= 1
#         # node.warn(f"Recognition results received. cntr {cntr}")
#         add_msg(obj_classification, 'dets', seq)#No seq?
#     sync_msgs = get_msgs()
#     if sync_msgs is not None:
#         img = sync_msgs['preview']
#         dets = sync_msgs['dets']
#         if -7 < cntr:
#             node.warn(f"NN too slow, skipping frame. Cntr {cntr}")
#             continue 
#         for i, det in enumerate(dets.detections):
#             if(det.label == 16):            
#                 cfg = ImageManipConfig()
#                 correct_bb(det)
#                 # # node.warn(det.label)
#                 # # node.warn(det.confidence)
#                 # #TODO ONLY DOG
#                 cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax) #Crop to keep only dogs
#                 # # node.warn(f"Sending {i + 1}. det. Seq {seq}. Det {det.label}, {det.ymin}, {det.xmax}, {det.ymax}")
#                 # node.warn(f"label{det.label}")

#                 cfg.setResize(shape,shape) #Resize for NN
#                 cfg.setKeepAspectRatio(False)
#                 # img = preproc(img,(shape,shape),mean,std)
#                 node.io['manip_cfg'].send(cfg)
#                 node.io['manip_img'].send(img)
#                 cntr += 1
# """)

# image_manip_node = pipeline.create(dai.node.ImageManip)
# # image_manip_node.initialConfig.setResize(224, 224)
# image_manip_node.setWaitForConfigInput(True)

# image_manip_script.outputs['manip_cfg'].link(image_manip_node.inputConfig)
# image_manip_script.outputs['manip_img'].link(image_manip_node.inputImage)

print("Creating recognition Neural Network...")
breed_nn = pipeline.create(dai.node.NeuralNetwork)
breed_nn.setBlobPath(str(Path("D:/Python/Luxonis/depthai-experiments/gen2-yolo/yolox/dogOnlineIP.blob").resolve().absolute()))
# breed_nn.setBlobPath(blobconverter.from_zoo(name="age-gender-recognition-retail-0013", shaves=6))
# image_manip_node.out.link(breed_nn.input) #Keep
breed_nn.input.setBlocking(True) #DEL

# Send converted frames from the host to the NN DEL
nn_xin = pipeline.create(dai.node.XLinkIn)
nn_xin.setStreamName("nnInput")
nn_xin.out.link(breed_nn.input)


breed_xout = pipeline.create(dai.node.XLinkOut) #To show breed on screen
breed_xout.setStreamName("recognition")
breed_nn.out.link(breed_xout.input)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    qRec = device.getOutputQueue(name="recognition", maxSize=4, blocking=False)


    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    shape = 224


    frame = None
    detections = []
    dogBreed = None
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
        

    while True:
        dogBreed = None
        if syncNN:
            inRgb = qRgb.get()
            inDet = qDet.get()
            inRec = qRec.get()
        else:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()
            inRec = qRec.tryGet()

        #---
        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)
            qNnInput = device.getInputQueue("nnInput", maxSize=10, blocking=False)
            #Get image , do everything and give to 2nd NN
            if inDet is not None:
                detections = inDet.detections
                counter += 1
                for det in detections:
                    if det.label ==DOG_INDEX:
                        dai_frame = dai.ImgFrame()
                        amplify_bb(det)
                        correct_bb(det)
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        #Crop based on det
                        # img_rgb_crop = AA #TODO
                        img_rgb = img_rgb[
                            int(det.ymin*img_rgb.shape[1]):int(det.ymax*img_rgb.shape[1]), 
                            int(det.xmin*img_rgb.shape[0]):int(det.xmax*img_rgb.shape[0]) ,
                            :
                            ]
                        img_rgb = cv2.resize(img_rgb, dsize=(shape, shape), interpolation=cv2.INTER_LINEAR)
                        image = preprocess_input(img_rgb)

                        # plt.imshow(image) #Needs to be in row,col order
                        # plt.axis('off')
                        # plt.savefig("image.png")
                        # print(np.max(image))
                        # print(np.min(image))

                        # NOTE: The model expects an FP16 input image, but ImgFrame accepts a list of ints only. I work around this by
                        # spreading the FP16 across two ints

                        #Try loading original image
                        loaded_arr = np.loadtxt("D:\Python\Luxonis\depthai-experiments\gen2-yolo\yolox\data.txt",delimiter=',')
                        load_original_arr = loaded_arr.reshape(
                            loaded_arr.shape[0], loaded_arr.shape[1] // 3, 3)
                        image = list(image.tobytes())
                        print(load_original_arr.shape)

                        dai_frame.setHeight(shape)
                        dai_frame.setWidth(shape)
                        dai_frame.setData(load_original_arr)
                        qNnInput.send(dai_frame)
        #----

        if inRec is not None:
            #TODO more than one
            # print(inRec.getSequenceNum())
            # print(type(inRec))
            # for iR in inRec:
            dogBreed = np.array(inRec.getLayerFp16('StatefulPartitionedCall/model_1/dense_3/Softmax'))
            #     # print(in_nn.getAllLayerNames())

        if frame is not None:
            color = (255, 0, 0)
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                if detection.label != DOG_INDEX:
                    cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"{int(detection.confidence *100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                else:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)#Only rectangle for dogs

                # print(dogBreeds)
                if dogBreed is not None:
                # if dogBreed is not None:#Can probably be deleted (if found dog there is always a breed associated)
                    whichDog = np.argmax(dogBreed, axis=0)
                    # print(np.sort(dogBreed,axis=None)[::-1])
                    if(dogBreed[whichDog]>0.8):
                        cv2.putText(frame, breedMap[whichDog].split("-")[-1], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(frame, f"{int(dogBreed[whichDog]*100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            # Show the frame
            cv2.imshow("RGB", frame)

        if cv2.waitKey(1) == ord('q'):
            break
