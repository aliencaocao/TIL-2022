import json

pred_path = "/content/drive/MyDrive/TIL 2022/CV/preds/cascade/cascade_pretrained_flip.bbox.json"
image_info = json.loads(open("/content/CV_data/test/interim_no_annotations.json").read())["images"]
preds = json.loads(open(pred_path).read())
filename_to_img = {}
for img in image_info:
  filename_to_img[img["file_name"]] = img
  filename_to_img[img["file_name"]]["has_pred"] = False

for filename in filename_to_img:
  if not filename_to_img[filename]["has_pred"]:
    # add dummy pred because DSTA grader gives error if one or more files has no preds
    preds.append({
      "image_id": filename_to_img[filename]["id"],
      "bbox": [0., 0., 0., 0.],
      "category_id": 2,
      "score": 0.0000001,
    })

preds = sorted(preds, key=lambda x: x['image_id'])

with open(pred_path, "w+") as all_preds_file:
  json.dump(preds, all_preds_file, separators=(",", ":"))
