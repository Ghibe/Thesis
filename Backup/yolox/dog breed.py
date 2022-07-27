from pathlib import Path
import numpy as np
import cv2
import depthai as dai
import time
from MultiMsgSync import TwoStageHostSeqSync

def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
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


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

syncNN = True

SHAPE = 416
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

p = dai.Pipeline()
p.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)


class FPSHandler:
    def __init__(self, cap=None):
        self.timestamp = time.time()
        self.start = time.time()
        self.frame_cnt = 0

    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)

camera = p.create(dai.node.ColorCamera)
camera.setPreviewSize(SHAPE, SHAPE)
camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camera.setInterleaved(False)
camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

nn = p.create(dai.node.YoloDetectionNetwork)
nn.setConfidenceThreshold(0.5)
nn.setIouThreshold(0.5)
nn.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
nn.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
nn.setBlobPath(str(Path("D:/Python/Luxonis/depthai-experiments/gen2-yolo/yolox/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob").resolve().absolute()))
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)
nn.setNumClasses(80)
nn.setCoordinateSize(4)
nn.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
nn.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})

# Send camera frames to the host
camera_xout = p.create(dai.node.XLinkOut)
camera_xout.setStreamName("camera")
camera.preview.link(camera_xout.input)

# Send converted frames from the host to the NN
nn_xin = p.create(dai.node.XLinkIn)
nn_xin.setStreamName("nnInput")
camera.preview.link(nn.input)#TODO

# Send bounding boxes from the NN to the host via XLink
nn_xout = p.create(dai.node.XLinkOut)
nn_xout.setStreamName("detection")
nn.out.link(nn_xout.input)

#Add node with new NN if is a dog
#TODO check accuracy dogs
#TODO check accuracy breeds
#----------------------------------------------------#

# # Script node will take the output from the face detection NN as an input and set ImageManipConfig
# # to the 'recognition_manip' to crop the initial frame
# image_manip_script = p.create(dai.node.Script)
# nn.out.link(image_manip_script.inputs['obj_in']) #Receive object

# # Remove in 2.18 and use `imgFrame.getSequenceNum()` in Script node
# nn.passthrough.link(image_manip_script.inputs['passthrough']) #bb

# nn_xin.out.link(image_manip_script.inputs['preview'])#Receive image

# image_manip_script.setScript("""
# import time
# msgs = dict()

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
#         add_msg(obj_classification, 'dets', seq)
#     sync_msgs = get_msgs()
#     if sync_msgs is not None:
#         img = sync_msgs['preview']
#         dets = sync_msgs['dets']
#         for i, det in enumerate(dets.detections): #NON ENTRA
#             node.warn("AA")
#             cfg = ImageManipConfig()
#             correct_bb(det)
#             # node.warn(det.label)
#             # node.warn(det.confidence)
#             #TODO ONLY DOG
#             cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax) #Crop to keep only dogs
#             # node.warn(f"Sending {i + 1}. det. Seq {seq}. Det {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
#             cfg.setResize(224, 224) #Resize for NN
#             cfg.setKeepAspectRatio(False)
#             node.io['manip_cfg'].send(cfg)
#             node.io['manip_img'].send(img)
#             #TODO PREPROCESSING IMAGE in teoria già in xin from host
# """)

# image_manip_node = p.create(dai.node.ImageManip)
# image_manip_node.initialConfig.setResize(224, 224)
# image_manip_node.setWaitForConfigInput(True)

# image_manip_script.outputs['manip_cfg'].link(image_manip_node.inputConfig)
# image_manip_script.outputs['manip_img'].link(image_manip_node.inputImage)

# print("Creating recognition Neural Network...")
# breed_nn = p.create(dai.node.NeuralNetwork)
# breed_nn.setBlobPath(str(Path("D:/Python/Luxonis/depthai-experiments/gen2-yolo/yolox/dog_breed_first_try_no_reverse_preprocess.blob").resolve().absolute()))#TODO
# # breed_nn.setBlobPath(blobconverter.from_zoo(name="age-gender-recognition-retail-0013", shaves=6))
# image_manip_node.out.link(breed_nn.input)

# breed_xout = p.create(dai.node.XLinkOut) #To show breed on screen
# breed_xout.setStreamName("recognition")
# breed_nn.out.link(breed_xout.input)









#-------------------------------------------------#
# Pipeline is defined, now we can connect to the device
with dai.Device(p) as device:
    sync = TwoStageHostSeqSync()
    queues = {}
    fps = FPSHandler()
    qCamera = device.getOutputQueue(name="camera", maxSize=10, blocking=False)
    qNnInput = device.getInputQueue("nnInput", maxSize=10, blocking=False)

    # Create output queues
    for name in ["detection"]:#, "recognition"]:
        queues[name] = device.getOutputQueue(name)

    # while True:
    #     for name, q in queues.items():
    #         # Add all msgs (color frames, object detections and recognitions) to the Sync class.
    #         if q.has():
    #             sync.add_msg(q.get(), name)
    # 
    # qNn = device.getOutputQueue(name="detection", maxSize=10, blocking=True)
    # breed = device.getOutputQueue(name="recognition", maxSize=10, blocking=True)

    while True:
        inRgb = qCamera.get()
        frame = inRgb.getCvFrame()
        # Set these according to your dataset OK!
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        image, ratio = preproc(frame, (SHAPE, SHAPE), mean, std)
        # NOTE: The model expects an FP16 input image, but ImgFrame accepts a list of ints only. I work around this by
        # spreading the FP16 across two ints
        image = list(image.tobytes())

        dai_frame = dai.ImgFrame()
        dai_frame.setHeight(SHAPE)
        dai_frame.setWidth(SHAPE)
        dai_frame.setData(image)
        qNnInput.send(dai_frame)

        for name, q in queues.items():
            # Add all msgs (color frames, object detections and recognitions) to the Sync class.
            if q.has():
                a = q.get()
                sync.add_msg(q.get(), name)
                for det in a.detections:
                    print(labelMap[det.label])
                    print({int(det.confidence *100)})
        msgs = sync.get_msgs()
        if msgs is not None:
            
        # if syncNN:
        #     in_nn = qNn.get()
        #     # breedRecognition = breed.get()
        # else:
        #     in_nn = qNn.tryGet()
        #     breedRecognition = breed.tryGet()
        # # print("Breed",breedRecognition) #Should be NN data
        # #TODO take breed[i] and do #TODO only get dogs??????
        # # np.squeeze(np.array(rec.getLayerFp16('prob')))#Then probably taxe max
        # print(in_nn)
        # if in_nn is not None:
            print("ENTER")
            fps.next_iter()
            cv2.putText(frame, "Fps: {:.2f}".format(fps.fps()), (2, SHAPE - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

            data = np.array(msgs["detection"].getLayerFp16('output')).reshape(1, 3549, 85)
            predictions = demo_postprocess(data, (SHAPE, SHAPE), p6=False)[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4, None] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.3)

            if dets is not None:
                final_boxes = dets[:, :4]
                final_scores, final_cls_inds = dets[:, 4], dets[:, 5]

                for i in range(len(final_boxes)):
                    bbox = final_boxes[i]
                    score = final_scores[i]
                    class_name = labelMap[int(final_cls_inds[i])].split("-")[-1]
                    # print(breedRecognition[i], "AA")
                    # dog_breed = np.squeeze(np.array(breedRecognition[i].getLayerFp16('prob')))#Layer name and type??
                    # dog_breed = breedMap[np.argmax(dog_breed)] #Get index of max value


                    if score >= 0.1:
                        # Limit the bounding box to 0..SHAPE
                        bbox[bbox > SHAPE - 1] = SHAPE - 1
                        bbox[bbox < 0] = 0
                        xy_min = (int(bbox[0]), int(bbox[1]))
                        xy_max = (int(bbox[2]), int(bbox[3]))
                        # Display detection's BB, label and confidence on the frame
                        cv2.rectangle(frame, xy_min , xy_max, (255, 0, 0), 2)
                        cv2.putText(frame, class_name, (xy_min[0] + 10, xy_min[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(frame, f"{int(score * 100)}%", (xy_min[0] + 10, xy_min[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        #TODO breed

        cv2.imshow("rgb", frame)
        if cv2.waitKey(1) == ord('q'):
            break
