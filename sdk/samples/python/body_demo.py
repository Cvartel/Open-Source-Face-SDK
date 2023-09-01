import sys
import argparse
import os.path

import cv2
import numpy as np

from face_sdk import Service

bone_map = [
    ("right_ankle", "right_knee"),
    ("right_knee", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_eye", "right_eye"),
    ("nose", "left_eye"),
    ("left_knee", "left_hip"),
    ("right_ear", "right_shoulder"),
    ("left_ear", "left_shoulder"),
    ("right_eye", "right_ear"),
    ("left_eye", "left_ear"),
    ("nose", "right_eye"),
    ("left_ankle", "left_knee")
]

def help_message():
    message = f"usage: {sys.argv[0]} [--mode detection | pose| reidentigication] " \
              " [--input_image <path to image>]" \
              " [--sdk_path ..]" \
              " [--output <yes/no>]"
    print(message)

def draw_bbox(obj, img, output, color=(0, 255, 0)):
    rect = obj["bbox"]
    if output == "yes":
        print(f"BBox coordinates: {int(rect[0].get_value() * img.shape[1])}, {int(rect[1].get_value() * img.shape[0])}, {(int(rect[2].get_value() * img.shape[1]))}, {int(rect[3].get_value() * img.shape[0])}")
    return cv2.rectangle(img, (int(rect[0].get_value() * img.shape[1]), int(rect[1].get_value() * img.shape[0])),
                         (int(rect[2].get_value() * img.shape[1]), int(rect[3].get_value() * img.shape[0])), color, 1)

def detector_demo(sdk_path, img_path, mode, output):
    service = Service.create_service(sdk_path)
    if not os.path.exists(img_path):
        raise Exception(f"not exist file {img_path}")

    body_detector = service.create_processing_block({"unit_type": "HUMAN_BODY_DETECTOR"})

    img: np.ndarray = cv2.imread(img_path, cv2.IMREAD_COLOR)
    input_image: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = input_image.shape

    imgCtx = {"blob": input_image.tobytes(), "dtype": "uint8_t", "format": "NDARRAY","shape": [dim for dim in input_image.shape]}
    ioData = service.create_context({"image": imgCtx})

    body_detector(ioData)
    if mode == "pose":
        pose_estimator = service.create_processing_block({"unit_type": "POSE_ESTIMATOR"})
        pose_estimator(ioData)

    elif mode == "reidentigication":
        reidentigication_detector = service.create_processing_block({"unit_type": "BODY_RE_IDENTIFICATION"})
        reidentigication_detector(ioData)
    color = (0,255,0)
    tink = 1

    for obj in ioData["objects"]:
        draw_bbox(obj, img, output)
        if mode == "pose":
            posesCtx = obj["keypoints"]
            for bone in bone_map:
                key1 = bone[0]
                key2 = bone[1]
                x1 = int(posesCtx[key1]["proj"][0].get_value() * img.shape[1])
                y1 = int(posesCtx[key1]["proj"][1].get_value() * img.shape[0])
                x2 = int(posesCtx[key2]["proj"][0].get_value() * img.shape[1])
                y2 = int(posesCtx[key2]["proj"][1].get_value() * img.shape[0])
                if output == "yes":
                    print("Pose: x1:",x1," y1:",y1," x2:",x2," y2:",y2)
                cv2.line(img,(x1,y1),(x2,y2),color,tink)

            for i in posesCtx.keys():
                x = int(posesCtx[i]["proj"][0].get_value()* img.shape[1])
                y = int(posesCtx[i]["proj"][1].get_value()* img.shape[0])
                cv2.circle(img,(x,y),3,(0,0,255),-1,0)

    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description='Video Recognition Demo')
    parser.add_argument('--mode', default="detection", type=str)
    parser.add_argument('--input_image', type=str, required=True)
    parser.add_argument('--sdk_path', default="../", type=str)
    parser.add_argument('--output', default="yes", type=str)

    return parser.parse_args()


if __name__ == "__main__":

    help_message()

    args = parse_args()

    if args.mode == "detection" or args.mode == "pose" or args.mode == "reidentigication":
        detector_demo(args.sdk_path, args.input_image, args.mode , args.output)
    else:
        print("Incorrect mode")