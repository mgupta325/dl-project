# Deep Learning project

CS 7643 Project Code

### Setup

1. Create python virtual env with python version >= 3.7 `python3.7 -m venv venv`
2. Enter venv `source venv/bin/avtivate`
3. First install Image library to avoid errors with other package installation `pip3 install Image`
4. Install remaining libraries `pip3 install -r requirements.txt`
5. Create models directory for Transformers model `mkdir models`
6. Download the final model weights from [here](https://drive.google.com/drive/folders/1Q7845UwHYhQztt_FUXlgRbuiTrVTV9No?usp=sharing)<br /> and place in models directory.
   NOTE: make sure CHECKPOINT variable in `transfomers_model.py` is the same as model weight file name, we use `model-embs4-seq80-auc0.7063-loss0.8723-acc0.6778`

### To recreate experiments

1. Download the dataset from [here](https://hatefulmemeschallenge.com/#download) and place the img folder in data/img.
2. Run Experiments notebooks using different training ratio jsonl files found in `data`, use different text encoders (commented out), use different slice image functions (commented out).
3. To see final model training see Training notebook.

### To use UI

1. To run the UI `python app.py`

The UI accepts images as input and returns the extracted text from the image and the classification of the image + text whether it is hateful or not and its probability.
