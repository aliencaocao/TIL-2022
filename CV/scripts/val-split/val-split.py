from pathlib import Path
import shutil
import random
import os

label_filenames = list(Path("C:/Users/yip/Downloads/CV_egg/labels").glob("*.txt"))
random.shuffle(label_filenames)

val_label_filenames = label_filenames[:int(0.2 * len(label_filenames))]

try:
  os.mkdir("C:/Users/yip/Downloads/CV_egg/val_images")
  os.mkdir("C:/Users/yip/Downloads/CV_egg/val_labels")
except:
  pass

for val_label_filename in val_label_filenames:
  val_image_filename = Path("C:/Users/yip/Downloads/CV_egg/images") / (val_label_filename.stem + ".png")
  shutil.move(str(val_image_filename), "C:/Users/yip/Downloads/CV_egg/val_images")
  shutil.move(str(val_label_filename), "C:/Users/yip/Downloads/CV_egg/val_labels")
