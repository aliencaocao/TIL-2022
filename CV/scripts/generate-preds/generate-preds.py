from pathlib import Path
import json

image_info = json.loads(open("interim_no_annotations.json").read())["images"]
filename_to_img = {}
for img in image_info:
  filename_to_img[img["file_name"]] = img
  filename_to_img[img["file_name"]]["has_pred"] = False

all_preds = []
for pred_txt_path in Path("/root/yolor/runs/detect/exp5/labels").glob("*.txt"):
  with open(str(pred_txt_path)) as pred_txt:
    for pred_line in pred_txt:
      filename = pred_txt_path.stem + (".jpg" if ".rf." in pred_txt_path.stem else ".png")
      img = filename_to_img[filename]

      class_idx, x, y, w, h, confidence = [float(x) for x in pred_line.split()]
      class_idx = int(class_idx)
      x = (x - w/2) * img["width"]
      y = (y - h/2) * img["height"]
      w *= img["width"]
      h *= img["height"]

      all_preds.append({
        "image_id": img["id"],
        # COCO labels are 1-indexed, and stupid me inverted the classes :)
        "category_id": 2 - class_idx,
        "bbox": [x, y, w, h],
        "score": confidence,
      })

      filename_to_img[filename]["has_pred"] = True

for filename in filename_to_img:
  if not filename_to_img[filename]["has_pred"]:
    # add dummy pred because DSTA grader gives error if one or more files has no preds
    all_preds.append({
      "image_id": filename_to_img[filename]["id"],
      "category_id": 2,
      "bbox": [0., 0., 0., 0.],
      "score": 0.01,
    })

with open("all_preds.json", "w") as all_preds_file:
  json.dump(all_preds, all_preds_file, separators=(",", ":"))
