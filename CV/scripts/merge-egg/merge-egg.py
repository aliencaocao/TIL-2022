import json

with open("easter_egg_day_2.json") as egg_json_file:
  egg_data = json.load(egg_json_file)
with open("train_egg1_merged.json") as train_json_file:
  train_data = json.load(train_json_file)

merged_data = train_data

old_id_to_new_id = {}
curr_img_id = 1 + max(img["id"] for img in train_data["images"])
for img in egg_data["images"]:
  new_img = img.copy()
  new_img["id"] = curr_img_id
  merged_data["images"].append(new_img)

  old_id_to_new_id[img["id"]] = new_img["id"]
  curr_img_id += 1

print(old_id_to_new_id)

curr_annotation_id = 1 + max(annotation["id"] for annotation in train_data["annotations"])
for annotation in egg_data["annotations"]:
  new_annotation = annotation
  new_annotation["id"] = curr_annotation_id
  new_annotation["image_id"] = old_id_to_new_id[annotation["image_id"]]
  new_annotation["category_id"] = 2 - annotation["category_id"] # egg labels are inverted relative to training data
  merged_data["annotations"].append(new_annotation)
  curr_annotation_id += 1

with open("train_egg2_merged.json", "w") as merged_json_file:
  json.dump(merged_data, merged_json_file, separators=(",", ":"))
