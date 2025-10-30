import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from rfdetr.util.coco_classes import COCO_CLASSES
from PIL import Image




def target_boxes(path_to_csv):
    '''
    Функция чтения bbox для тестовой выборки из .сsv файла
    '''

    df = pd.read_csv(path_to_csv)
    filenames = df['filename'].unique().tolist()

    gt_by_img = {}
    for _, row in df.iterrows():
        fname = str(row['filename'])
        bbox = [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
        gt_by_img.setdefault(fname, []).append(bbox)

    # Считаем количество bbox
    total_gts = sum(len(v) for v in gt_by_img.values())
    print(f"Уникальных изображений в CSV: {len(gt_by_img)}, общее число bbox: {total_gts}")

    return gt_by_img




def check_class(model):
    '''
    Функция проверки модели на наличие класса person
    '''

    person_class_idx = None
    names = model.names
    for k, v in names.items():
        if str(v).lower() == "person":
            person_class_idx = int(k)
            break

    print(f"Найден индекс класса 'person' = {person_class_idx} (в model.names).")

    return person_class_idx




def check_class_rtdetr(model):
    '''
    Функция проверки модели на наличие класса person
    '''

    for k, v in COCO_CLASSES.items():
        if str(v).lower() == "person":
            person_class_idx = int(k)
            break
        
    print(f"person_class_idx = {person_class_idx}")

    return person_class_idx




def iou_xyxy(boxA, boxB):
    '''
    Расчет IoU
    '''

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea




def clip_box(box, w, h):
    '''
    Нормализация bbox (по размеру изображения)
    '''

    x1 = max(0, min(w-1, int(box[0])))
    y1 = max(0, min(h-1, int(box[1])))
    x2 = max(0, min(w-1, int(box[2])))
    y2 = max(0, min(h-1, int(box[3])))
    return [x1, y1, x2, y2]




def map50_calculate(predictions, targets, iou_treshhold):
    preds_sorted = sorted(predictions, key=lambda x: x['score'], reverse=True)

    matched_gt = {img: np.zeros(len(targets.get(img, [])), dtype=bool) for img in targets.keys()}

    tp_list = []
    fp_list = []

    for pred in preds_sorted:
        img = pred['image_id']
        pred_box = pred['bbox']
        max_iou = 0.0
        max_iou_idx = -1
        gts = targets.get(img, [])

        for idx, gt_box in enumerate(gts):
            if matched_gt[img][idx]:
                continue
            iou_val = iou_xyxy(pred_box, gt_box)
            if iou_val > max_iou:
                max_iou = iou_val
                max_iou_idx = idx
        if max_iou >= iou_treshhold:
            tp_list.append(1)
            fp_list.append(0)
            matched_gt[img][max_iou_idx] = True
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    total_gts = sum(len(v) for v in targets.values())

    precisions = tp_cum / (tp_cum + fp_cum + 1e-12)
    recalls = tp_cum / (total_gts + 1e-12)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])

    i_list = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i-1]) * mpre[i]

    print(f"mAP50 для датасета = {ap:.4f}")




def read_reference_video(model, video_path, save_video_path, person_class_idx):
    """
    Видео-инференс для YOLO.
    """
        
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(save_video_path, fourcc, fps, (w,h))

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    pbar = tqdm(total=total_frames, desc="Video inference", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)
        if len(results) > 0:
            r = results[0]
            boxes = getattr(r.boxes, "xyxy", None)
            scores = getattr(r.boxes, "conf", None)
            classes = getattr(r.boxes, "cls", None)
            if boxes is not None:
                for i in range(len(boxes)):
                    cls = int(classes[i].item()) if classes is not None else None
                    if cls != person_class_idx:
                        continue
                    xyxy = boxes[i].cpu().numpy().astype(int).tolist()
                    score = float(scores[i].cpu().numpy().tolist()) if scores is not None else 1.0
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0,255,0), 2)
                    cv2.putText(frame, f"person {score:.2f}", (xyxy[0], max(0, xyxy[1]-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        out.write(frame)
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    out.release()
    print(f"Видео с боксами сохранено в {save_video_path}")




def read_reference_video_rfdetr(model, video_path, save_video_path, person_class_idx, conf_thresh=0.25):
    """
    Видео-инференс для RF-DETR.
    """

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(save_video_path, fourcc, fps, (w, h))

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    pbar = tqdm(total=total_frames, desc="Video inference", unit="frame")

    frame_idx = 0
    total_drawn = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = model.predict(frame, threshold=float(conf_thresh))
        det = detections[0] if isinstance(detections, (list, tuple)) else detections

        if not hasattr(det, "xyxy") or len(det.xyxy) == 0:
            continue

        img_w, img_h = frame.shape[1], frame.shape[0]
    
        xyxys = np.array(det.xyxy)
        scores = np.array(det.confidence)
        classes = np.array(det.class_id)

        for box, score, cls in zip(xyxys, scores, classes):
            cls_int = int(cls)
            if cls_int != person_class_idx:
                continue

            xyxy = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
            xyxy = clip_box(xyxy, img_w, img_h)
            cv2.rectangle(frame, 
                          (xyxy[0], xyxy[1]), 
                           (xyxy[2], xyxy[3]), 
                           (0,255,0), 
                           2)
            cv2.putText(frame, 
                        f"person {score:.2f}", 
                        (xyxy[0], max(0, xyxy[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0,255,0), 
                        1)
            total_drawn += 1

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print(f"Видео с боксами сохранено в {save_video_path}. Кадров обработано: {frame_idx}, отрисовано bbox: {total_drawn}")