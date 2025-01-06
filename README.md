# Beverage Container Detection Using YOLO


This project demonstrates how to train, validate, and use a **YOLOv11** model to **detect** 
**beverage containers** using a dataset from **Roboflow**. YOLO (You Only Look Once) is a state-of-
the-art object detection model known for its speed and accuracy, making it ideal for real-time 
applications.


## Table of Contents

1. [Project Overview](https://github.com/elnemr19/Beverage-Container-Detection-Using-YOLO/tree/main?tab=readme-ov-file#project-overview)

2. [Dataset](https://github.com/elnemr19/Beverage-Container-Detection-Using-YOLO/tree/main?tab=readme-ov-file#dataset)

3. [Setup and Installation](https://github.com/elnemr19/Beverage-Container-Detection-Using-YOLO/tree/main?tab=readme-ov-file#setup-and-installation)

4. [Model Training](https://github.com/elnemr19/Beverage-Container-Detection-Using-YOLO/tree/main?tab=readme-ov-file#model-training)

5. [Validation](https://github.com/elnemr19/Beverage-Container-Detection-Using-YOLO/tree/main?tab=readme-ov-file#validation)

6. [Prediction](https://github.com/elnemr19/Beverage-Container-Detection-Using-YOLO/tree/main?tab=readme-ov-file#prediction)

7. [Results](https://github.com/elnemr19/Beverage-Container-Detection-Using-YOLO/tree/main?tab=readme-ov-file#results)



## Project Overview

This project aims to create a robust YOLOv11 model capable of detecting beverage 
containers in images. By leveraging Roboflow for dataset preparation and YOLOv11 for training, 
the project achieves high accuracy in detecting various types of containers.



## Dataset

The dataset is sourced from [Roboflow](https://universe.roboflow.com/30201220/beverage-containers-3atxb-vqovx-1ubgt/dataset/1/download), containing labeled images of beverage containers. The dataset includes training, validation, and test sets for optimal model performance.

**Key Details:**

* **Number of Classes:** Multiple beverage container types.

* **Image Resolution:** 640x640 (optimized for YOLOv11).

* **Format:** YOLO-specific annotations.


To download the dataset, we used the following Roboflow API integration:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("roboflow-universe-projects").project("beverage-containers-3atxb")
version = project.version(3)
dataset = version.download("yolov11")
```



## Setup and Installation

1. Install the necessary dependencies:
```python
  pip install roboflow ultralytics
```
2. Integrate the dataset using the Roboflow API and download the YOLOv11 configuration file.

3. Ensure the YOLOv11 environment is ready:
 ```python
  yolo check
 ```



## Model Training

To train the YOLOv11 model, execute the following command:

```python
!yolo task=detect mode=train model=yolo11s.pt data={dataset.location}/data.yaml epochs=50 imgsz=640 plots=True
```

**Training Parameters**:

* **Model:** YOLOv11 small (yolo11s.pt)

* **Epochs:** 50

* **Image Size:** 640x640

* **Plots:** Enabled for visualization of training progress.




## Validation

After training, validate the model to evaluate its performance:

```python
!yolo task=detect mode=val model=/{HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml
```

**Validation Metrics**:

* **mAP (mean Average Precision)**: Indicates detection accuracy.

* **Precision and Recall**: Evaluate the balance between true positives and false negatives.



## Prediction

Use the trained model to detect beverage containers in new images:

```python
!yolo task=detect mode=predict model=/{HOME}/runs/detect/train/weights/best.pt conf=0.3 source={dataset.location}/test/images
```

**Parameters**:

* **Confidence Threshold:** 0.3

* **Source**: Test images from the dataset.

The output will include:

* Bounding boxes around detected objects.

* Class labels and confidence scores.

* Annotated images saved in the designated output directory.

## Evaluation

**Confusion Matrix**

![conv](https://github.com/user-attachments/assets/98c0a5bb-1151-43da-8f14-946783b4cb99)



![image](https://github.com/user-attachments/assets/0702b24e-eb38-4a3e-b6f3-b6a213d09a3b)




## Results

The trained YOLOv11 model achieves high accuracy in detecting beverage containers. Below are sample detections from the test set:


![image](https://github.com/user-attachments/assets/04bc1ae2-0026-4511-99bf-1a7fb9f502e4)



## Kaggle Link: [Kaggle](https://www.kaggle.com/code/abdullahalnemr/beverage-containers-yolo#Evaluation)

