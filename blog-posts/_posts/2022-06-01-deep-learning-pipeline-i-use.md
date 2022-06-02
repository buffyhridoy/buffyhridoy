---
layout: post
title: Deep Learning Pipeline I Use(and Model Interpretability using Grad CAM)
description: >
  In this blog, I have showed the pipeline I use for deep learning 
canonical_url: http://Deep Learning Pipeline I Use(and Model Interpretability using Grad CAM)
hide_image: false
accent_color: '#4fb1ba'
accent_image:
  background: 'linear-gradient(to bottom,#193747 0%,#233e4c 30%,#3c929e 50%,#d5d5d4 70%,#cdccc8 100%)'
  overlay:    true
---

For this we have used a oct dataset [dataset](https://www.kaggle.com/datasets/paultimothymooney/kermany2018/) which contain 4 classes.
We have done this whole project on a kaggle notebook.

## Import packages
```python
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from tensorflow.keras.applications.vgg16 import VGG16
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D,AveragePooling2D, BatchNormalization, PReLU, ReLU
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications import ResNet50, InceptionResNetV2
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report
from keras.preprocessing.image import load_img,img_to_array
from PIL import Image
import matplotlib.cm as cm

```

## Set the path
```python
train_path = "/kaggle/input/kermany2018/OCT2017 /train"
val_path = "/kaggle/input/kermany2018/OCT2017 /val"
test_path = "/kaggle/input/kermany2018/OCT2017 /test"
```

## Create X_train, Y_train, X_test, Y_test
```python
X_train = []
Y_train = []

for target in os.listdir(train_path):
    target_path = os.path.join(train_path, target)
    for file in tqdm(os.listdir(target_path)):
        file_path = os.path.join(target_path, file)
        X_train.append(file_path)
        Y_train.append(target)
```
![image](https://user-images.githubusercontent.com/37147511/171443142-6afa2b69-3933-4bc2-a817-608c82c66ab4.png)

    
```python
for target in os.listdir(val_path):
    target_path = os.path.join(val_path, target)
    for file in tqdm(os.listdir(target_path)):
        file_path = os.path.join(target_path, file)
        X_train.append(file_path)
        Y_train.append(target)
```
![image](https://user-images.githubusercontent.com/37147511/171443246-91eaa241-5794-40ac-99cb-e4b2e4911563.png)

```python
X_test = []
Y_test = []

for target in os.listdir(test_path):
    target_path = os.path.join(test_path, target)
    for file in tqdm(os.listdir(target_path)):
        file_path = os.path.join(target_path, file)
        X_test.append(file_path)
        Y_test.append(target)
```
![image](https://user-images.githubusercontent.com/37147511/171443322-8bff10d7-9f13-4f9e-84bc-784d06e77c71.png)

## Train and Validation split
```python
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
```

## Countplot (Y_train)
```python
sns.countplot(x = Y_train)
```
![image](https://user-images.githubusercontent.com/37147511/171444173-e4ddff9d-d596-4d29-b909-5168f80f158d.png)

## Countplot (Y_val)
```python
sns.countplot(x = Y_val)
```
![image](https://user-images.githubusercontent.com/37147511/171444348-e3f5c036-d581-4ddf-b716-3e57492482c8.png)


## Countplot (Y_test)
```python
sns.countplot(x = Y_test)
```
![image](https://user-images.githubusercontent.com/37147511/171444477-8aed98fc-66ef-40a6-8fbd-ff070557a19b.png)

## Create DataFrame
```python
df_train = pd.DataFrame(list(zip(X_train, Y_train)), columns =['image_path', 'label'])
df_val = pd.DataFrame(list(zip(X_val, Y_val)), columns =['image_path', 'label'])
df_test = pd.DataFrame(list(zip(X_test, Y_test)), columns =['image_path', 'label'])
```

## DataGenerator
```python
from keras.preprocessing.image import ImageDataGenerator

train_aug = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    rescale = 1./255,
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)

test_aug = ImageDataGenerator(
    rescale = 1./255,
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)

train_generator= train_aug.flow_from_dataframe(
    dataframe=df_train,
    x_col="image_path",
    y_col="label",
    batch_size=16,
    color_mode="rgb",
    target_size = (224, 224),
    class_mode="categorical")

val_generator= test_aug.flow_from_dataframe(
    dataframe=df_val,
    x_col="image_path",
    y_col="label",
    batch_size=16,
    color_mode="rgb",
    target_size = (224, 224),
    class_mode="categorical")

test_generator= test_aug.flow_from_dataframe(
    dataframe=df_test,
    x_col="image_path",
    y_col="label",
    color_mode="rgb",
    batch_size=16,
    shuffle = False, 
    target_size = (224, 224),
    class_mode="categorical")
```
![image](https://user-images.githubusercontent.com/37147511/171445300-3be5ca03-24d7-47d7-a4e8-6014e96d577f.png)


## Define generate_model Function
```python
def generate_model(pretrained_model = 'vgg16', num_classes=4):
    if pretrained_model == 'inceptionv3':
        base_model = InceptionV3(weights = 'imagenet', include_top=False, input_shape=(224, 224, 3))
    elif pretrained_model == 'inceptionresnet':
        base_model = InceptionResNetV2(weights = 'imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        base_model = VGG16(weights = 'imagenet', include_top=False, input_shape=(224, 224, 3)) # Topless
    
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(4096)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    #Freezing Convolutional Base
    for layer in base_model.layers[:-3]:
        layer.trainable = False  
    return model
```

## Define train_model Function
```python
def train_model(model, train_generator, test_generator, num_epochs, optimizer, metrics):
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer, 
                  metrics=metrics)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=6, verbose=1)
    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=7)
    print(model.summary())
    
    history = model.fit(train_generator, epochs=num_epochs, 
                        validation_data=test_generator, verbose=1,
                        callbacks = [early_stop, rlr])
    
    return model, history
```
## metrics
```python
metrics = ['accuracy',
                tf.keras.metrics.AUC(),
                tfa.metrics.CohenKappa(num_classes = 4),
                tfa.metrics.F1Score(num_classes = 4),
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall()]
```

## Define Different plot function(plot_loss, plot_acc, plot_confusion_matrix, plot_roc_curves)
```python
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
# It prints & plots the confusion matrix, normalization can be applied by setting normalize=True.
    
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):

    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_roc_curves(y_true, y_pred, num_classes, class_labels):
    
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_test = lb.transform(y_true)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    for i in range(num_classes):
        fig, c_ax = plt.subplots(1,1, figsize = (6, 4))
        c_ax.plot(fpr[i], tpr[i],
                 label='ROC curve of class {0} (area = {1:0.4f})'
                 ''.format(class_labels[i], roc_auc[i]))
        c_ax.set_xlabel('False Positive Rate')
        c_ax.set_ylabel('True Positive Rate')
        c_ax.set_title('ROC curve of class {0}'.format(class_labels[i]))
        c_ax.legend(loc="lower right")
        plt.show()
    return roc_auc_score(y_test, y_pred)
```
## Define evaluate_model Function
```python
def evaluate_model(model, history, test_generator):
    # Evaluate model
    score = model.evaluate(test_generator, verbose=0)
    print('\nTest set accuracy:', score[1], '\n')
    
    y_true = np.array(test_generator.labels)
    y_pred = model.predict(test_generator, verbose = 1)
    y_pred_classes = np.argmax(y_pred,axis = 1)
    class_labels = list(test_generator.class_indices.keys())   
    
    print('\n', sklearn.metrics.classification_report(y_true, y_pred_classes, target_names=class_labels), sep='')
    confusion_mtx = confusion_matrix(y_true, y_pred_classes)
    plot_acc(history)
    plt.show()
    plot_loss(history)
    plt.show()
    plot_confusion_matrix(confusion_mtx, classes = class_labels)
    plt.show()
    print("ROS AUC score:", plot_roc_curves(y_true, y_pred, 4, class_labels))
```

## VGG model
```python
vgg_model = generate_model('vgg16', 4)
```
![image](https://user-images.githubusercontent.com/37147511/171447904-b1db45b1-7ca2-4c6b-8bcb-ac2ec2a4a31a.png)


## Train the model
```python
vgg_model, vgg_history = train_model(vgg_model, train_generator, val_generator, 10, tf.keras.optimizers.SGD(lr=0.001, momentum=0.9), metrics)
```
![image](https://user-images.githubusercontent.com/37147511/171550609-12d1c252-0c58-41e8-ba83-359c68a4ff3c.png)

## Evaluate Model
```python
evaluate_model(vgg_model, vgg_history, test_generator)
```
![image](https://user-images.githubusercontent.com/37147511/171550729-d8fd1c18-fa4c-41fb-a8b1-2071710e38e7.png)
![image](https://user-images.githubusercontent.com/37147511/171550771-08f7717a-c240-49d1-8f83-954b41ed9faa.png)
![image](https://user-images.githubusercontent.com/37147511/171550840-d8cd066a-7179-4526-8b74-1d94a3776b1c.png)
![image](https://user-images.githubusercontent.com/37147511/171550911-78f11677-12cf-4a4b-9a60-2fe81f05c3dd.png)
![image](https://user-images.githubusercontent.com/37147511/171550955-61142cab-dd1c-48f5-ae9e-3cd682c1621d.png)
![image](https://user-images.githubusercontent.com/37147511/171551073-f4059d79-7ba5-4478-959b-bc66274be73d.png)
![image](https://user-images.githubusercontent.com/37147511/171551138-714bb2c3-96e6-4e09-92f4-d7e863131c0c.png)
![image](https://user-images.githubusercontent.com/37147511/171551171-65d844e3-3c56-49ca-98b4-f31f69476c6d.png)

# Model Interpretability using Grad CAM
One way to ensure the model is performing correctly is to debug your model and visually validate that it is “looking” and “activating” at the correct locations in an image.

Selvaraju et al. published a novel paper entitled, Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization (https://arxiv.org/abs/1610.02391).

GRAD-CAM works by (1) finding the final convolutional layer in the network and then (2) examining the gradient information flowing into that layer.

The output of Grad-CAM is a heatmap visualization for a given class label (either the top, predicted label or an arbitrary label we select for debugging). We can use this heatmap to visually verify where in the image the CNN is looking.

```python
image_path= df_test['image_path'][101]

img = load_img(image_path, target_size=(224,224,3)) # stores image in PIL format
image_array=img_to_array(img)
```
## Display Original Image 
```python
from PIL import Image
display(Image.open(image_path))
```
![image](https://user-images.githubusercontent.com/37147511/171556737-8ff42afd-010b-4bf2-a0c3-c1d57ee68d9f.png)

## GRAD CAM Algorithm
```python
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.layers[-2].output])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
```
## Create Heat Map
```python
# Make model
model = vgg_model
last_conv_layer_name ="block5_conv3"
# Remove last layer's softmax
model.layers[-1].activation = None

img_array=np.expand_dims(image_array, axis=0)
# Prepare particular image 

# Generate class activation heatmap
heatmap= make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# Display heatmap
plt.matshow(heatmap)
plt.show()
```
![image](https://user-images.githubusercontent.com/37147511/171557041-3e46fc87-9dfd-43c0-977b-afea70400ece.png)

```python
def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image.open(cam_path))
```
```python
save_and_display_gradcam(image_path, heatmap,cam_path="/kaggle/working/GradCamTest.jpg")
```
![image](https://user-images.githubusercontent.com/37147511/171557542-e3dc3393-b8b6-4c30-a06a-cd03e44f9ad5.png)

