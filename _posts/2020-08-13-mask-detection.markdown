---
title : "Face mask detection alert system"
date : 2020-08-13
tags : [machine learning, computer vision, CNN, python, keras, opencv]
header :
  image : "./assets/images/facemaskdetection/facemaskdetection.jpg"
excerpt : "Machine learning , computer vision, Covid19"
mathjax : true
---
# Face Mask detection alert system
*The objective of this project is to develop an alert system that detects if a person is wearing a mask or not and triggers a notification through the video management system. This is a case of deep learning convolution neural network CNN, where the model is first trained on a set of faces with and without mask, and then used to classify new data.*
## Introduction

The coronavirus pandemic has pushed people across the world into difficult times and uncertainty. For my part I was wondering how I can play my part to fight against coronavirus. I decided to contribute and started thinking about digital solutions that I can develop from my home. I came up with this project that consists in detecting Face Mask,  using existing IP cameras and CCTV cameras combined with Computer Vision to detect people without masks and to trigger a notification.

As several countries on the continent saw an uptick in reported cases of COVID-19, they moved to make mask-wearing mandatory in public and private spaces. Face mask detecting alert systems help authorities to control people entering public and private places that rnow required to wear face coverings.
### Example of application
##### Airports
The Face Mask Detection System can be used at airports to detect travelers without masks. Face data of travelers can be captured in the system at the entrance. If a traveler is found to be without a face mask, their picture is sent to the airport authorities so that they could take quick action. If the person’s face is already stored, like the face of an Airport worker, it can send the alert to the worker’s phone directly.
##### Hospitals
Using Face Mask Detection System, Hospitals can monitor if their staff is wearing masks during their shift or not. If any health worker is found without a mask, they will receive a notification with a reminder to wear a mask. Also, if quarantine people who are required to wear a mask, the system can keep an eye and detect if the mask is present or not and send notification automatically or report to the authorities.
##### Offices
The Face Mask Detection System can be used at office premises to detect if employees are maintaining safety standards at work. It monitors employees without masks and sends them a reminder to wear a mask. The reports can be downloaded or sent an email at the end of the day to capture people who are not complying with the regulations or the requirements.

## Body
## ---
In order to train a custom face mask detector, we need to break our project into two distinct phases, each with its own respective sub-steps :

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/facemaskdetection/face_mask_detection_phases.png" alt="">

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/facemaskdetection/face_mask_detection_phases.png.jpg)
*Fig. 1:Phases and individual steps for building a face mask detector with computer vision and deep learning using Python, OpenCV, and TensorFlow/Keras.*

1- Training: consists in loading our face mask detection dataset from disk, training a model (using Keras/TensorFlow) on this dataset, and then serializing the face mask detector to disk.
<br>
2- Deployment: Once the face mask detector is trained, we can then move on to loading the mask detector, performing face detection, and then classifying each face as with_mask or without_mask

## Data Set and Processing

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/facemaskdetection/withandwhithoutmask.jpeg" alt="">
*Fig. 2: Dataset seperated into two files : images of faces with mask and images of faces without mask*

Data Section - Include written descriptions of data and follow with relevant spreadsheets.
Methods Section - Explain how you gathered and analyzed data.
Analysis Section - Explain what you analyzed. Include any charts here.
Results - Describe the results of your analysis.
## Conclusion
Restate the questions from your introduction.
Restate important results.
Include any recommendations for additional data as needed.

## Appendix

Include the details of your data and process here.
Include any secondary data, including references.


{% if page.mathjax %}
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
{% endif %}