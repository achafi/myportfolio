---
title : "Face mask detection alert system"
date : 2020-08-13
tags : [machine learning, computer vision, CNN, python, keras, opencv]
header :
  image : "./assets/images/facemaskdetection/Business-parks-and-offices.jpg"
excerpt : "Machine learning , computer vision, Covid19"
mathjax : true
---
# Face Mask detection alert system
*The objective of this project is to develop an alert system that detects if a person is wearing a mask or not and triggers a notification through the video management system. This is a case of deep learning convolution neural network CNN, where the model is first trained on a set of faces with and without mask, and then used to classify new data.*
[Github : Source code](https://github.com/achafi/FaceMaskDectionAlertSystem)
## Introduction

The coronavirus pandemic has pushed people across the world into difficult times and uncertainty. For my part I was wondering how I can play my part to fight against coronavirus. I decided to contribute and started thinking about digital solutions that I can develop from my home. I came up with this project that consists in detecting Face Mask,  using existing IP cameras and CCTV cameras combined with Computer Vision to detect people without masks and to trigger a notification.

As several countries on the continent saw an uptick in reported cases of COVID-19, they moved to make mask-wearing mandatory in public and private spaces. Face mask detecting alert systems help authorities to control people entering public and private places that rnow required to wear face coverings.
### Example of application
- The Face Mask Detection System can be used at airports to detect travelers without masks. Face data of travelers can be captured in the system at the entrance. If a traveler is found to be without a face mask, their picture is sent to the airport authorities so that they could take quick action. If the person’s face is already stored, like the face of an Airport worker, it can send the alert to the worker’s phone directly.
- Using Face Mask Detection System, Hospitals can monitor if their staff is wearing masks during their shift or not. If any health worker is found without a mask, they will receive a notification with a reminder to wear a mask. Also, if quarantine people who are required to wear a mask, the system can keep an eye and detect if the mask is present or not and send notification automatically or report to the authorities.
- The Face Mask Detection System can be used at office premises to detect if employees are maintaining safety standards at work. It monitors employees without masks and sends them a reminder to wear a mask. The reports can be downloaded or sent an email at the end of the day to capture people who are not complying with the regulations or the requirements.

## The flow and main phases of the entire system

1- Person appears at entrance, wearing mask or not wearing mask
2- Camera detects the mask on the Person's face
3- Person denied Access until he/she wears Mask
4- Authorities are alerted via Email in real time

In order to train a custom face mask detector, we need to break our project into two distinct phases, each with its own respective sub-steps :

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/facemaskdetection/face_mask_detection_phases.png" alt="">
*Fig. 1:Phases and individual steps for building a face mask detector with computer vision and deep learning using Python, OpenCV, and TensorFlow/Keras.*

1- Training: consists in loading our face mask detection dataset from disk, training a model (using Keras/TensorFlow) on this dataset, and then serializing the face mask detector to disk.
<br>
2- Deployment: Once the face mask detector is trained, we can then move on to loading the mask detector, performing face detection, and then classifying each face as with_mask or without_mask

## Data Set and Processing

Our [Data](https://github.com/achafi/FaceMaskDectionAlertSystem/tree/master/Dataset) consisted of **917 images**:
  - with_mask  : **451 images** face images
  - without_mask : **466 images** without masks
<br>
The mask are artificially added to the images in order to have a dataset images where a person is wearing a mask and not.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/facemaskdetection/withandwhithoutmask.jpeg" alt="">
*Fig. 2: Dataset seperated into two files : images of faces with mask and images of faces without mask*

We first convert images to Grayscale and separate out labels and images using the OpenCV packages for Python: cv2
```python
# Converting the image into gray scale
  grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# resizing the gray scaled image into size 56*56 in order to keep size of the images consistent
  resized_img = cv2.resize(grayscale_img, (img_rows, img_cols))
```
Then we perform one hot encoding on the labels since the label are in textual form, for that we use scikit learn preprocessing module.

```python
# one hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
labels = np.array(labels)
```
## Build Convolutional neural network
After preprocessing the images, it is time to build a Convolutional Neural Network using Sequential API of Keras.This model aims to classify whether an image is of face with mask or without mask
<br>
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/facemaskdetection/cnn_model.png" alt="">
*Fig. 3: architecture of CNN model*
<br>
As shown above in figure 3, the first layer group contains Convolution, Relu and MaxPooling layers. The second layer group contains Convolution, Relu and MaxPooling layers. we then add a flatten and Dropout Layer to stack the output convolutions as well as cater overfitting. Last but not least we add a Relu activation and softmax classifier.

### Compile and train the model
Once the Keras model is defined, it’s time to decide the initial hyperparameters.
- **Optimizer** : Adam with an initial learning rate of lr = 0.001
- **Loss function** : categorical_crossentropy that computes the crossentropy loss between the labels and predictions
- **Metric** : Accuracy

```python
from keras.optimizers import Adam
epochs = 50
model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics = ['accuracy'])
fitted_model = model.fit(
    train_X,
    train_y,
    epochs = epochs,
    validation_split=0.25)
```

### Model Evaluation
Results : loss: 0.0048 - accuracy: 0.9977 - val_loss: 0.2708 - val_accuracy: 0.9456
<br>
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/facemaskdetection/loss.png" alt="">
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/facemaskdetection/accuraccy.png" alt="">
*Fig. 4: Plot the Training Loss & Accuracy*
<br>
We notice that the model is overfitted since the gap of training and validation loss is not minimal. Some improvements should be expected like :
- Adding more data.
- Data augmentation.
- Search for better architectures that generalize well
- Adding regularization (mostly dropout, L1/L2 regularization are also possible)

## Deployment
We use the Live Webcam Video stream to Detect the Face, then we extract the region of interest of the Face with cascad classifier.The next step is to engage trained our pretrained CNN Face Mask Detection Model to the face identified and determine if the person is wearing Mask or Not.
<br>
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/facemaskdetection/cascade_classifier.PNG" alt="">
*Fig. 5: Cascad classifier*
<br>
```python
# Classifier to detect face
face_det_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#  Capture Video
vid_source=cv2.VideoCapture(0)
```
The system will trigger a warning Message in terms of pop up window to highlight that access is denied if the person has not worn the face mask. In this case an email or SMS could be send to the concerned person or authorities. No access will be given until the person wears the mask.

```python
# If label = 1 then it means wearing No Mask and 0 means wearing Mask
        if (label == 1):
          # Pop up message

          messagebox.showwarning("Warning","Access Denied. Please wear a Face Mask")

          # Send an email to the administrator if access denied/user not wearing face mask
          message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT)
          mail = smtplib.SMTP('smtp.gmail.com', 587)
          mail.ehlo()
          mail.starttls()
          mail.login('assia.chafii93@gmail.com','*******')
          mail.sendmail('assia.chafii93@gmail.com','assia.chafii93@gmail.com',message)
          mail.close
        else:
          pass
          break
```

## Conclusion

In this project, I built a Face Mask Alert system detector using Convolutional Neural Networks (CNN) Python, Keras, Tensorflow and OpenCV. With further improvements these types of models could be integrated with CCTV or other types cameras to detect and identify people without masks. With the prevailing worldwide situation due to COVID-19 pandemic, these types of systems would be very supportive for many kind of institutions around the world.

{% if page.mathjax %}
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
{% endif %}