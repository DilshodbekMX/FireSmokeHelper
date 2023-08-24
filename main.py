import datetime
import os
import threading
import pytz
import urllib.request
import cv2
import settings
from yolov8 import YOLOv8

db = settings.FIRESTORE_DB
storage = settings.FIREBASE_BUCKET
file_num = 0
SECONDS_TO_COUNTDOWN = 5
countdown = SECONDS_TO_COUNTDOWN
countdown_timestamp = cv2.getTickCount()
camera_urls, camera_names, user_ids, online_user_ids, camera_ids = [], [], [], [], []

callback_done = threading.Event()


def file_exists(filepath):
    try:
        f = open(filepath, 'r')
        exists = True
        f.close()
    except:
        exists = False
    return exists


def get_filepath(save_path, file_name, file_suffix):
    global file_num
    file_path = save_path + file_name + str(file_num) + file_suffix
    while file_exists(file_path):
        file_num += 1
        file_path = save_path + file_name + str(file_num) + file_suffix

    return file_path


def on_snapshot(col_snapshot, changes, read_time):
    for doc in col_snapshot:
        dict_doc = doc.to_dict()
        camera_ids.append(doc.id)
        camera_urls.append(dict_doc["web_address"])
        user_ids.append(dict_doc["offline_user_id"])
        camera_names.append(dict_doc["camera_name"])
        online_user_ids.append(dict_doc["online_user_id"])
    callback_done.set()


col_query = db.collection("Cameras")
docs = db.collection("Cameras").stream()
for doc in docs:
    dict_doc = doc.to_dict()
    camera_ids.append(doc.id)
    camera_urls.append(dict_doc["web_address"])
    user_ids.append(dict_doc["offline_user_id"])
    camera_names.append(dict_doc["camera_name"])
    online_user_ids.append(dict_doc["online_user_id"])
query_watch = col_query.on_snapshot(on_snapshot)


def upload_to_local():
    file_suffix = ".jpg"
    model_path = settings.MEDIA_ROOT + "/yolo_model/"
    save_path = settings.MEDIA_ROOT + "/captured_images/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    global countdown_timestamp, countdown
    i = 0
    print(i)
    while True:
        print(i)
        print((len(camera_urls)))
        if len(camera_urls) <= 0:
            break
        if i % len(camera_urls) == 0 and i >= 1000:
            i = 0
        name = camera_urls[i % len(camera_urls)]
        print(name)
        if urllib.request.urlopen(name).getcode() == 200:
            vid = cv2.VideoCapture(name)
            while True:
                timestamp = cv2.getTickCount()
                ret, frame = vid.read()
                if (timestamp - countdown_timestamp) / cv2.getTickFrequency() > 1.0:
                    file_name = str(user_ids[i % len(camera_urls)]) + "#" + camera_names[i % len(camera_urls)] + "#"
                    countdown_timestamp = cv2.getTickCount()
                    countdown -= 1
                    print(countdown)
                    if countdown <= 0:
                        countdown = SECONDS_TO_COUNTDOWN
                        try:
                            model_file = model_path + "best.onnx"
                            yolov8_detector = YOLOv8(model_file, conf_thres=0.2, iou_thres=0.3)
                            boxes, scores, class_ids = yolov8_detector(frame)
                            if len(boxes) > 0 and len(scores) > 0 and len(class_ids) > 0:
                                print("Detected image")
                                detected_classes = []
                                offline_user_id = user_ids[i % len(camera_urls)]
                                online_user_id = online_user_ids[i % len(camera_urls)]
                                camera_name = camera_names[i % len(camera_urls)]
                                current_time = datetime.datetime.now(pytz.timezone('Asia/Tashkent'))
                                for box, score, class_id in zip(boxes, scores, class_ids):
                                    if class_id == 0:
                                        detected_classes.append(f"Smoke {round(score * 100, 2)} %")
                                    else:
                                        detected_classes.append(f"Fire {round(score * 100, 2)} %")
                                combined_img = yolov8_detector.draw_detections(frame)
                                cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
                                filepath = get_filepath(save_path, file_name, file_suffix)
                                is_saved = cv2.imwrite(filepath, combined_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
                                if is_saved:
                                    print(f"Image saved to {filepath}")
                                    blob = storage.blob(f"{camera_name}#{current_time}", chunk_size=262144)
                                    blob.upload_from_filename(filepath)
                                    blob.make_public()
                                    new_doc = db.collection("Detected").document()
                                    print(detected_classes)
                                    print(offline_user_id)
                                    print(online_user_id)
                                    print(camera_name)
                                    print(current_time)
                                    data = {
                                        "detected_classes": detected_classes,
                                        "camera_name": camera_name,
                                        "offline_user_id": offline_user_id,
                                        "online_user_id": online_user_id,
                                        "id": new_doc.id,
                                        "detected_time": current_time,
                                        "image_url": blob.public_url,
                                    }
                                    print(data)
                                    new_doc.set(data)
                                    os.remove(filepath)
                            else:
                                print("No objects detected")
                        except:
                            break
                if cv2.waitKey(10) == ord('q'):
                    break
            i += 1
            vid.release()
            cv2.destroyAllWindows()



upload_to_local()
