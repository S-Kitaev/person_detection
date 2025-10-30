from ultralytics import YOLO
from rfdetr import RFDETRBase
from datetime import datetime

from inference_functions.yolo_inf_functions import read_reference_video, read_reference_video_rfdetr

# Список исследованных моделей
models = {
    "yolo11": YOLO("models/weights/1_yolo11n_base.pt"),
    "yolo12": YOLO("models/weights/2_yolo12n_base.pt"),
    "yolo_user": YOLO("models/weights/3_yolo_users.pt"),
    "rt-detr": RFDETRBase()
}

# Функция для создания видео с выделением людей
def process_video(model_name, 
                  path_to_video, 
                  path_to_save="reference video/detected_persons/new_detections"):

    model = models[model_name]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path_to_save_video =path_to_save + "/" + model_name + current_time + "detected.mp4"
    if model_name != "rt-detr":
        person_class_idx = 0
        read_reference_video(model, path_to_video, path_to_save_video, person_class_idx)

    else:
        person_class_idx = 1
        read_reference_video_rfdetr(model, path_to_video, path_to_save_video, person_class_idx)


# Пример использования функции
process_video("yolo11", "reference video/crowd.mp4")