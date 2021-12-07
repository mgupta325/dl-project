import pandas as pd
import random

df = pd.read_json("data/train.jsonl", lines=True)

indices = [i for i in range(8500)]
random.shuffle(indices)

id, img, label, text = [],[],[],[]
label_0, label_1 = 0, 0 
total_images = 500
label_0_total = int(total_images * 0.5)
label_1_total = int(total_images * 0.5)



for idx in indices:
  row = df.iloc[idx]
  if (row["label"] == 0 and label_0 < label_0_total):
    id.append(row["id"])
    img.append(row["img"])
    label.append(row["label"])
    text.append(row["text"])
    label_0 += 1
  elif (row["label"] == 1 and label_1 < label_1_total):
    id.append(row["id"])
    img.append(row["img"])
    label.append(row["label"])
    text.append(row["text"])
    label_1 += 1
  
  if len(id) == 2000:
    break

split_df = pd.DataFrame(
          {'id': id,
          'img': img,
          'label': label,
          'text': text
          })

split_df.to_json("data/small_50_50_split.jsonl", orient='records', lines=True)






