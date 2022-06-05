import json
from pathlib import Path
from ensemble_boxes import nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion

def unnormalized_xywh_to_normalized_xyxy(unnormalized_xywh, img_width, img_height):
  x, y, w, h = unnormalized_xywh
  return [
    x / img_width,
    y / img_height,
    (x + w) / img_width,
    (y + h) / img_height,
  ]

def normalized_xyxy_to_unnormalized_xywh(normalized_xyxy, img_width, img_height):
  x1, y1, x2, y2 = normalized_xyxy
  return [
    x1 * img_width,
    y1 * img_height,
    (x2 - x1) * img_width,
    (y2 - y1) * img_height,
  ]

# we assume these dirs are folders of pred txt files from different models
yolo_txt_dir_paths = [
  "../generate-preds/labels",
]

img_info = json.loads(open("../generate-preds/interim_no_annotations.json").read())["images"]
filename_to_img = {}
for img in img_info:
  filename_to_img[img["file_name"]] = img
  filename_to_img[img["file_name"]]["has_pred"] = False

all_models_preds = {}
for txt_dir_path in yolo_txt_dir_paths:
  all_models_preds[txt_dir_path] = {}

  for txt_path in Path(txt_dir_path).glob("*.txt"):
    filename = txt_path.stem + (".jpg" if ".rf." in txt_path.stem else ".png")
    img = filename_to_img[filename]

    all_models_preds[txt_dir_path][img["id"]] = []
    with open(str(txt_path)) as txt_file:
      for pred_line in txt_file:
        class_idx, x, y, w, h, confidence = [float(x) for x in pred_line.split()]
        class_idx = int(class_idx)
        x = (x - w/2) * img["width"]
        y = (y - h/2) * img["height"]
        w *= img["width"]
        h *= img["height"]

        all_models_preds[txt_dir_path][img["id"]].append({
          "image_id": img["id"],
          # COCO labels are 1-indexed, and stupid me inverted the classes :)
          "category_id": 2 - class_idx,
          "bbox": [x, y, w, h],
          "score": confidence,
        })

wbf_all_preds = []
img_has_pred = {}
for img in img_info:
  boxes_list = []
  scores_list = []
  labels_list = []
  weights = []

  for model in all_models_preds:
    if img["id"] in all_models_preds[model]:
      this_model_preds = all_models_preds[model][img["id"]]
      boxes_list.append([unnormalized_xywh_to_normalized_xyxy(pred["bbox"], img["width"], img["height"]) for pred in this_model_preds])
      scores_list.append([pred["score"] for pred in this_model_preds])
      labels_list.append([pred["category_id"] for pred in this_model_preds])
      weights.append(1) # all models have same weightage

  boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights)
  for box, score, label in zip(boxes, scores, labels):
    wbf_all_preds.append({
      "image_id": img["id"],
      "bbox": normalized_xyxy_to_unnormalized_xywh(box, img["width"], img["height"]),
      "category_id": label,
      "score": score,
    })

  img_has_pred[img["id"]] = len(boxes) > 0

for img_id in img_has_pred:
  if not img_has_pred[img_id]:
    wbf_all_preds.append({
      "image_id": img_id,
      "bbox": [0., 0., 0., 0.],
      "category_id": 2,
      "score": 0.01,
    })

with open(f"wbf_({', '.join(all_models_preds).replace('/', '-')}).json", "w") as wbf_preds_file:
  json.dump(wbf_all_preds, wbf_preds_file, separators=(",", ":"))
