---
title : "Covid-19 Detection from X-Ray"
date : 2020-07-01
tags : [machine learning, Opencv, CNN, Python, Keras]
header:
  video:
    id: YvjVpde2DN8
    provider: youtube
excerpt : "Machine learning , CNN, Python"
mathjax : true
---
☝️ [to the video demo on my Youtube channel](https://youtu.be/YvjVpde2DN8)
[Source code](https://github.com/achafi/Covid19Detector)

# Automated detection of COVID-19 cases using deep neural networks with X-ray images
*The objective of this project is to develop a web application that provides an accurate diagnosis of COVID-19 from chest X-ray images. I created my own deep learning CNN model for early detection of COVID-19, then deployed it using Flask in order to be available for the end-users so they can make use of it.*

## Introduction
The novel coronavirus 2019 (COVID-2019), which first appeared in Wuhan city of China in December 2019, spread rapidly around the world and became a pandemic.It has caused a devastating effect on both daily lives, public health, and the global economy. It is critical to detect the positive cases as early as possible so as to prevent the further spread of this epidemic and to quickly treat affected patients.The need for auxiliary diagnostic tools has increased as there are no accurate automated toolkits available. Recent findings obtained using radiology imaging techniques suggest that such images contain salient information about the COVID-19 virus. Application of advanced artificial intelligence (AI) techniques coupled with radiological imaging can be helpful for the accurate detection of this disease, and can also be assistive to overcome the problem of a lack of specialized physicians in remote villages.
<br>

## Inspiration
*Why do we need a deep learning model for early Covid-19 Detection?*
Because of the followings :
-	Blood Tests are costly
-	Blood Tests take time to conduct ~5 hours per patient
-	Extent of Spread can be detected
-	Defaillant or inaccurate toolkits

## Data Collection

At the time I decided to start this project, I didn't find ready open-source data of x-ray images of both positive and negative patients. For normal x-ray images, I find Kaggle [Chest X-ray dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia?). For affected chest, i keep searching till i found out a [github repository](https://github.com/ieee8023/covid-chestxray-dataset/tree/master/images) that shares an open dataset of chest X-ray and CT images of patients which are positive or suspected of COVID-19 or other viral and bacterial pneumonias (MERS, SARS, and ARDS.).
All the data is collected from public sources as well as through indirect collection from hospitals and physicians.
<br>
Creation of Covid-19 samples from the github images repository and Metadata.csv.
Metadata.csv contains extra data about the patient (age, sex....) and the label of the findings ('COVID-19', 'ARDS', 'SARS', 'Pneumocystis', 'Streptococcus','No Finding', 'Chlamydophila', 'E.Coli'....).For our purpose, we read only images of COVID-19 positive, so we should filter rows from Metadat.csv to keep only rows where["finding"]== "COVID-19" with a front view.

```python
# To Create data for positive samples
FILE_PATH = "images/metadata.csv"
IMAGES_PATH = "images"
# Shortlis of covid-19 xray images with front view (PA)
cnt = 0

for (i, row) in df.iterrows() :
    if row["finding"]== "COVID-19" and row['view']=="PA":
        filename = row["filename"]
        image_path = os.path.join(IMAGES_PATH, filename)
        image_copy_path = os.path.join(TARGET_DIR, filename)
        shutil.copy2(image_path, image_copy_path)
        print("Moving image", cnt)
        cnt += 1
```
[The integral data creation code](https://github.com/achafi/Covid19Detector/blob/master/DataSet%20Creator.ipynb)
Our Final dataset is composed of :
- Covid positive Samples : **142 samples**
- Normal Chest X-Rays : **142 samples**
## Convolutional neural network architecture
After preprocessing the images, it is time to build a Convolutional Neural Network using Sequential API of Keras.This model aims to classify whether an image covid-19 positive or negative sample.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/xraycoviddetecting/covid19-xray-detection-architecture.png" alt="">
*Fig. 1: CNN model architecture*

As shown above in figure 1, the first layer group contains Convolution, Relu and MaxPooling layers. The second layer group contains Convolution, Relu and MaxPooling layers. I then add a flatten and Relu activation layer to stack the output convolutions as well as cater overfitting. Last but not least I add a Sigmoid classifier.


<br>
To be continued ....

## Technology and tools used for this project covers

- Python
- Numpy and Pandas for data prerocessing
- Matplotlib for data visualization
- Keras sequential model for CNN building
- Jupyter notebook, visual studio code as IDE
- Python flask for http server
- HTML/CSS/Javascript for UI

{% if page.mathjax %}
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
{% endif %}