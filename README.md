# ASL-gesture-recognizer
Tensorflow implementation of Inception-Resnet V2 for recognizing American Sign Language gestures.

## Requirements:
- TensorFlow (version 2.0+)
- OpenCV
- Numpy
- gTTS (For GUI)
- Tkinter (For GUI)

## Instructions:
1. Clone the repository using:
```
git clone https://github.com/SarkarAnuragParth/ASL-gesture-recognizer.git
```
2. Download the dataset from https://www.kaggle.com/datasets/ayuraj/asl-dataset.
3. Train the model using:
```
python main.py \
--train_path ./path to data directory \
--mode train \
--epochs 20
```
4. After training, you can test the model using:
   ```
   python main.py \
   --train_path ./path to data directory \
   --mode test \


   
This implementation features a GUI which you can upload images into. The output will be relayed via text as well as speech. 
Please note that the GUI requires an active internet connection to use the gTTS module.
To start the GUI, run the following script:
```
python guiofasl.py
```



