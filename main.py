import datetime
import os
import threading
import time
import pytz
import urllib.request
import cv2
import settings
from PIL import Image
from yolov8 import YOLOv8

db = settings.FIRESTORE_DB
storage = settings.FIREBASE_BUCKET
file_num = 0  # Save images to current directory
SECONDS_TO_COUNTDOWN = 2
SECONDS_TO_COUNTDOWN2 = 30
countdown = SECONDS_TO_COUNTDOWN
countdown2 = SECONDS_TO_COUNTDOWN2
countdown_timestamp = cv2.getTickCount()
camera_urls = []
camera_names = []
user_ids = []
camera_ids = []
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


def get_filepath2(save_path, file_name, file_suffix):
    global file_num
    file_path = save_path + file_name + file_suffix
    while file_exists(file_path):
        file_num += 1
        file_path = save_path + file_name + file_suffix
    return file_path


def on_snapshot(col_snapshot, changes, read_time):
    for doc in col_snapshot:
        dict_doc = doc.to_dict()
        camera_ids.append(doc.id)
        camera_urls.append(dict_doc["web_address"])
        user_ids.append(dict_doc["offline_user_id"])
        camera_names.append(dict_doc["camera_name"])
    callback_done.set()


col_query = db.collection("Cameras")
query_watch = col_query.on_snapshot(on_snapshot)


def upload_to_local():
    file_suffix = ".jpg"
    save_path = settings.MEDIA_ROOT + "/" + "captured_images/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    global countdown_timestamp, countdown
    i = 0
    while True:
        res = [file_path for file_path in os.listdir(save_path) if
               os.path.isfile(os.path.join(save_path, file_path))]
        if len(camera_urls) <= 0:
            break
        name = camera_urls[i % len(camera_urls)]
        if urllib.request.urlopen(name).getcode() == 200:
            vid = cv2.VideoCapture(name)
            while vid:
                timestamp = cv2.getTickCount()
                ret, frame = vid.read()
                if (timestamp - countdown_timestamp) / cv2.getTickFrequency() > 1.0:
                    file_name = str(user_ids[i % len(camera_urls)]) + "#" + camera_names[i % len(camera_urls)] + "#"
                    countdown_timestamp = cv2.getTickCount()
                    countdown -= 1
                    if countdown <= 0:
                        try:
                            filepath = get_filepath(save_path, file_name, file_suffix)
                            cv2.imwrite(filepath, frame)
                            countdown = SECONDS_TO_COUNTDOWN
                        except:
                            pass
                        break
                if cv2.waitKey(10) == ord('q'):
                    break
            i += 1
            vid.release()
            cv2.destroyAllWindows()
            print(f"Captured Resources length: {len(res)}")
            if len(res) >= 60:
                print("Stopped")
                break


def get_and_detect():
    file_suffix = ".jpg"
    save_path = settings.MEDIA_ROOT + "/detected_images/"
    model_path = settings.MEDIA_ROOT + "/yolo_model/"
    source_path = settings.MEDIA_ROOT + "/captured_images/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(source_path):
        os.makedirs(source_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    res = [file_path for file_path in os.listdir(source_path) if
           os.path.isfile(os.path.join(source_path, file_path))]
    if len(res) > 0:
        for image in res:
            if os.path.isfile(os.path.join(source_path, image)):
                model_file = model_path + "best.onnx"
                yolov8_detector = YOLOv8(model_file, conf_thres=0.2, iou_thres=0.3)
                img_url = os.path.join(source_path, image)
                img = cv2.imread(img_url)
                boxes, scores, class_ids = yolov8_detector(img)
                if len(boxes) > 0 and len(scores) > 0 and len(class_ids) > 0:
                    print("Detected image")
                    detected_classes = []
                    for box, score, class_id in zip(boxes, scores, class_ids):
                        if class_id == 0:
                            detected_classes.append(f"Smoke {round(score * 100, 2)} %")
                        else:
                            detected_classes.append(f"Fire {round(score * 100, 2)} %")
                    img_list = image.split("#")
                    offline_user_id = img_list[0]
                    camera_name = img_list[1]
                    new_doc = db.collection("Detected").document()
                    data = {
                        "detected_classes": detected_classes,
                        "camera_name": camera_name,
                        "offline_user_id": offline_user_id,
                        "id": new_doc.id,
                        "detected_time": datetime.datetime.now(pytz.timezone('Asia/Tashkent')),
                    }
                    new_doc.set(data)
                    new_image_name = f"{new_doc.id}#{camera_name}"
                    combined_img = yolov8_detector.draw_detections(img)
                    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
                    filepath = get_filepath2(save_path, new_image_name, file_suffix)
                    cv2.imwrite(filepath, combined_img)
                os.remove(os.path.join(source_path, image))
            else:
                break


def upload_to_cloud():
    source_path = settings.MEDIA_ROOT + "/detected_images/"
    res = [file_path for file_path in os.listdir(source_path) if
           os.path.isfile(os.path.join(source_path, file_path))]
    if len(res) > 0:
        for image in res:
            if os.path.isfile(os.path.join(source_path, image)):
                print("Detected image saved to cloud")
                img_list = image.split("#")
                doc_id = img_list[0]
                camera_name = img_list[1]
                detected_ref = db.collection("Detected").document(doc_id)
                current_time = datetime.datetime.now(pytz.timezone('Asia/Tashkent'))
                blob = storage.blob(f"Detected/{camera_name}#{current_time}")
                blob.upload_from_filename(os.path.join(source_path, image))
                blob.make_public()
                data = {
                    "image_url": blob.public_url,
                    "detected_time": current_time,
                }
                detected_ref.update(data)
                os.remove(os.path.join(source_path, image))
            else:
                break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    while True:
        print("upload_to_local: ")
        upload_to_local()
        print("get_and_detect: ")
        get_and_detect()
        print("upload_to_cloud: ")
        upload_to_cloud()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
