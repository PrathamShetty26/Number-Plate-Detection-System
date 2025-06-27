import os
import cv2
import numpy as np
from skimage import io
import easyocr

# import matplotlib.pyplot as plt
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

net = cv2.dnn.readNetFromONNX('./yolov5/runs/train/Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def get_detections(img, net):
    # 1.CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # 2. GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections


def non_maximum_supression(input_image, detections):
    # 3. FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE

    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]  # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5]  # probability score of license plate
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)

    # 4.1 CLEAN
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    # 4.2 NMS
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)

    return boxes_np, confidences_np, index


def drawings(image, boxes_np, confidences_np, index):
    # 5. Drawings
    texts = []
    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)
        license_text = extract_text(image, boxes_np[ind])
        texts.append(license_text)

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 0, 255), -1)
        cv2.rectangle(image, (x, y + h), (x + w, y + h + 25), (0, 0, 0), -1)

        cv2.putText(image, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(image, license_text, (x, y + h + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    return image, texts


# predictions flow with return result
def yolo_predictions(img, net):
    # step-1: detections
    input_image, detections = get_detections(img, net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    # step-3: Drawings
    result_img, texts = drawings(img, boxes_np, confidences_np, index)
    return result_img, texts





# initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

def extract_text(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y + h, x:x + w]

    if 0 in roi.shape:
        return 'no number'

    else:
        # extract text using EasyOCR
        result = reader.readtext(roi)
        text = ' '.join([res[1] for res in result])
        text = text.strip()

        return text




def text_correction(text):
    text = text.upper()
    text = text.replace(" ","")
    text = text.replace(".", "")
    text = text.replace(",","")
    if(len(text)>8):
        # alphabet - number
        text = text[:2] + text[2:4].replace("S", "5") + text[4:6] + text[6:].replace("S", "5")
        text = text[:2] + text[2:4].replace("l", "1") + text[4:6] + text[6:].replace("l", "1")
        text = text[:2] + text[2:4].replace("i", "1") + text[4:6] + text[6:].replace("i", "1")
        text = text[:2] + text[2:4].replace("I", "1") + text[4:6] + text[6:].replace("I", "1")
        text = text[:2] + text[2:4].replace("B", "8") + text[4:6] + text[6:].replace("B", "8")
        text = text[:2] + text[2:4].replace("J", "1") + text[4:6] + text[6:].replace("J", "1")
        text = text[:2] + text[2:4].replace("A", "4") + text[4:6] + text[6:].replace("A", "4")
        text = text[:2] + text[2:4].replace("G", "9") + text[4:6] + text[6:].replace("G", "9")
        text = text[:2] + text[2:4].replace("O", "0") + text[4:6] + text[6:].replace("O", "0")
        text = text[:2] + text[2:4].replace("Z", "2") + text[4:6] + text[6:].replace("Z", "2")
        text = text[:2] + text[2:4].replace("E", "3") + text[4:6] + text[6:].replace("E", "3")
        text = text[:2] + text[2:4].replace("G", "6") + text[4:6] + text[6:].replace("G", "6")
        text = text[:2] + text[2:4].replace("T", "7") + text[4:6] + text[6:].replace("T", "7")
        
        # number - alphabet
        text = text[:2].replace("5", "S") + text[2:4] + text[4:6].replace("5", "S") + text[6:]
        text = text[:2].replace("1", "I") + text[2:4] + text[4:6].replace("1", "I") + text[6:]
        text = text[:2].replace("4", "A") + text[2:4] + text[4:6].replace("4", "A") + text[6:]
        text = text[:2].replace("9", "G") + text[2:4] + text[4:6].replace("9", "G") + text[6:]
        text = text[:2].replace("0", "O") + text[2:4] + text[4:6].replace("0", "O") + text[6:]
        text = text[:2].replace("2", "Z") + text[2:4] + text[4:6].replace("2", "Z") + text[6:]
        text = text[:2].replace("3", "E") + text[2:4] + text[4:6].replace("3", "E") + text[6:]
        text = text[:2].replace("8", "B") + text[2:4] + text[4:6].replace("8", "B") + text[6:]
        text = text[:2].replace("6", "G") + text[2:4] + text[4:6].replace("6", "G") + text[6:]
        text = text[:2].replace("7", "T") + text[2:4] + text[4:6].replace("7", "T") + text[6:]
        
        
        # symbol - alphabet
        text = text[:2].replace("&", "Q") + text[2:4] + text[4:6].replace("&", "Q") + text[6:]
        text = text[:2].replace("§", "S") + text[2:4] + text[4:6].replace("§", "S") + text[6:]
        text = text[:2].replace("°", "O") + text[2:4] + text[4:6].replace("°", "O") + text[6:]
        text = text[:2].replace("©", "C") + text[2:4] + text[4:6].replace("©", "C") + text[6:]
        text = text[:2].replace("®", "R") + text[2:4] + text[4:6].replace("®", "R") + text[6:]
        
        #symbol - number
        text = text[:2] + text[2:4].replace("!", "1") + text[4:6] + text[6:].replace("!", "1")
        text = text[:2] + text[2:4].replace("∞", "8") + text[4:6] + text[6:].replace("∞", "8")
        text = text[:2] + text[2:4].replace("]", "1") + text[4:6] + text[6:].replace("]", "1")
        text = text[:2] + text[2:4].replace("[", "1") + text[4:6] + text[6:].replace("[", "1")
    print(text)
    return text


def is_valid_plate_syntax(plate):
#     return (len(plate) > 8 and all(ch.isalpha() for ch in plate[:2]) and all(ch.isnumeric() for ch in plate[2:4]) and all(ch.isalpha() for ch in plate[4:6]) and all(ch.isnumeric() for ch in plate[6:]))
    plate = plate.replace(" ", "")
    res = len(plate) >= 6 and len(plate) <= 10 and plate[0:2].isalpha() and plate[2:4].isdigit();
    length = len(plate)
    i = 4
    cnt = 0
    while i < length and plate[i].isalpha():
        cnt += 1
        i += 1
    res = res and cnt >= 1 and cnt <= 2
    cnt = 0
    while i < length and plate[i].isdigit():
        cnt += 1
        i += 1
    res = res and cnt >= 1 and cnt <= 4 and i == length
    return res
def valid_plates(results):
    result_set = set()
    for texts in results:
        for text in texts:
            text = text_correction(text)
            if (is_valid_plate_syntax(text)):
                result_set.add(text)
    return result_set

def read_video(path):
    text_results = []

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            print('Unable to read video')
            break

        for i in range(int(fps/5)):
            ret = cap.grab()

        results, texts = yolo_predictions(frame, net)
        text_results.append(texts)
        print(texts)
    return list(valid_plates(text_results))


def read_img(path):
    img = io.imread(path)
    results, texts = yolo_predictions(img, net)
    # print(texts)
    return texts
