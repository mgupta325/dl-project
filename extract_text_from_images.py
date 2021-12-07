import numpy as np
import pandas as pd
from PIL import Image
from OCR import get_text

df = pd.read_json("data/test_unseen.jsonl", lines=True)
df['ocr_text']=""
did_not_work = []


for idx,row in df.iterrows():
  print(f"Evaluation #: {idx}")
  im = Image.open("data/"+row['img'])
  arr = np.asarray(im)
  try:
    text = get_text(arr)
  except:
    did_not_work.append(row["img"])
    text = ""
  row['ocr_text'] = text
  df.iloc[idx] = row

print(did_not_work)
df.to_json("data/test_unseen_with_extracted_text.jsonl", orient='records', lines=True)
