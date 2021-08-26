import argparse
from argparse import ArgumentParser
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
 
# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
 
model_path = os.path.join('infer-model', 'resnet50_csv_56.h5')
 
# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')
 
# load label to names mapping for visualization purposes
labels_to_names = {
    0:"No entry",
    1:"No parking",
    2:"No turning",
    3:"Max Speed 40",
    4:"Max Speed 50",
    5:"Max Speed 60",
    6:"Max Speed 70",
    7:"Max Speed 80",
    8:"Other prohibition",
    9:"Warning",
    10:"Mandatory",
    11:"Indication" ,
}
 
# load image
image = read_image_bgr('images/3728.png')
 
# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
 
# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image,626,1622)
 
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
 
# correct for image scale
boxes /= scale
 
# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    
    # scores are sorted so we can break
    if score < 0.7:
        break
 
    color = label_color(label)

    #if label > -1:
      #print(labels_to_names[label])
 
    b = box.astype(int)
    draw_box(draw, b, color=color)
 
    caption = "{} {:.5f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)
 
plt.figure(figsize=(10, 5))
plt.axis('off')
plt.imshow(draw)
plt.tight_layout()
plt.show()
