---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.6.4
  nbformat: 4
  nbformat_minor: 4
---

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:27:37.945764Z\",\"shell.execute_reply.started\":\"2022-06-23T10:27:37.946155Z\",\"iopub.status.idle\":\"2022-06-23T10:27:45.589904Z\",\"iopub.execute_input\":\"2022-06-23T10:27:37.946187Z\",\"shell.execute_reply\":\"2022-06-23T10:27:45.588725Z\"}" trusted="true"}
``` {.python}
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
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:27:45.591762Z\",\"shell.execute_reply.started\":\"2022-06-23T10:27:45.592165Z\",\"iopub.status.idle\":\"2022-06-23T10:27:45.597963Z\",\"iopub.execute_input\":\"2022-06-23T10:27:45.592209Z\",\"shell.execute_reply\":\"2022-06-23T10:27:45.596717Z\"}" trusted="true"}
``` {.python}
train_path = "../input/oct-dataset-duke-srinivasan-2014"
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:27:45.60055Z\",\"shell.execute_reply.started\":\"2022-06-23T10:27:45.601241Z\",\"iopub.status.idle\":\"2022-06-23T10:27:46.035296Z\",\"iopub.execute_input\":\"2022-06-23T10:27:45.601334Z\",\"shell.execute_reply\":\"2022-06-23T10:27:46.03416Z\"}" trusted="true"}
``` {.python}
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

X_train = []
Y_train = []

for target in os.listdir(train_path):
    target_path = os.path.join(train_path, target)
    for file in tqdm(os.listdir(target_path)):
        file_path = os.path.join(target_path, file)
        X_train.append(file_path)
        Y_train.append(target)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:27:46.038187Z\",\"shell.execute_reply.started\":\"2022-06-23T10:27:46.03868Z\",\"iopub.status.idle\":\"2022-06-23T10:27:46.261448Z\",\"iopub.execute_input\":\"2022-06-23T10:27:46.038727Z\",\"shell.execute_reply\":\"2022-06-23T10:27:46.260247Z\"}" trusted="true"}
``` {.python}
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:27:46.263324Z\",\"shell.execute_reply.started\":\"2022-06-23T10:27:46.263889Z\",\"iopub.status.idle\":\"2022-06-23T10:27:46.271452Z\",\"iopub.execute_input\":\"2022-06-23T10:27:46.263938Z\",\"shell.execute_reply\":\"2022-06-23T10:27:46.270236Z\"}" trusted="true"}
``` {.python}
X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=0.5, random_state=42)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:27:46.273005Z\",\"shell.execute_reply.started\":\"2022-06-23T10:27:46.2734Z\",\"iopub.status.idle\":\"2022-06-23T10:27:46.490806Z\",\"iopub.execute_input\":\"2022-06-23T10:27:46.27352Z\",\"shell.execute_reply\":\"2022-06-23T10:27:46.489623Z\"}" trusted="true"}
``` {.python}
sns.countplot(x = Y_train)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:27:46.492417Z\",\"shell.execute_reply.started\":\"2022-06-23T10:27:46.492841Z\",\"iopub.status.idle\":\"2022-06-23T10:27:46.660181Z\",\"iopub.execute_input\":\"2022-06-23T10:27:46.492884Z\",\"shell.execute_reply\":\"2022-06-23T10:27:46.658862Z\"}" trusted="true"}
``` {.python}
sns.countplot(x = Y_val)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:27:46.664369Z\",\"shell.execute_reply.started\":\"2022-06-23T10:27:46.664811Z\",\"iopub.status.idle\":\"2022-06-23T10:27:46.998594Z\",\"iopub.execute_input\":\"2022-06-23T10:27:46.664848Z\",\"shell.execute_reply\":\"2022-06-23T10:27:46.99752Z\"}" trusted="true"}
``` {.python}
sns.countplot(x = Y_test)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:27:47.001175Z\",\"shell.execute_reply.started\":\"2022-06-23T10:27:47.001624Z\",\"iopub.status.idle\":\"2022-06-23T10:27:47.013555Z\",\"iopub.execute_input\":\"2022-06-23T10:27:47.001704Z\",\"shell.execute_reply\":\"2022-06-23T10:27:47.012075Z\"}" trusted="true"}
``` {.python}
df_train = pd.DataFrame(list(zip(X_train, Y_train)), columns =['image_path', 'label'])
df_val = pd.DataFrame(list(zip(X_val, Y_val)), columns =['image_path', 'label'])
df_test = pd.DataFrame(list(zip(X_test, Y_test)), columns =['image_path', 'label'])
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:27:47.015415Z\",\"shell.execute_reply.started\":\"2022-06-23T10:27:47.016187Z\",\"iopub.status.idle\":\"2022-06-23T10:27:47.673494Z\",\"iopub.execute_input\":\"2022-06-23T10:27:47.016231Z\",\"shell.execute_reply\":\"2022-06-23T10:27:47.672174Z\"}" trusted="true"}
``` {.python}
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
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:27:47.675177Z\",\"shell.execute_reply.started\":\"2022-06-23T10:27:47.675593Z\",\"iopub.status.idle\":\"2022-06-23T10:27:47.687677Z\",\"iopub.execute_input\":\"2022-06-23T10:27:47.675656Z\",\"shell.execute_reply\":\"2022-06-23T10:27:47.686318Z\"}" trusted="true"}
``` {.python}
from tensorflow.keras.applications.vgg16 import VGG16
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D,AveragePooling2D, BatchNormalization, PReLU, ReLU
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications import ResNet50, InceptionResNetV2

def generate_model(pretrained_model = 'vgg16', num_classes=3):
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
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:27:47.689632Z\",\"shell.execute_reply.started\":\"2022-06-23T10:27:47.690491Z\",\"iopub.status.idle\":\"2022-06-23T10:27:47.70628Z\",\"iopub.execute_input\":\"2022-06-23T10:27:47.69068Z\",\"shell.execute_reply\":\"2022-06-23T10:27:47.70485Z\"}" trusted="true"}
``` {.python}
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
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:27:47.708096Z\",\"shell.execute_reply.started\":\"2022-06-23T10:27:47.708528Z\",\"iopub.status.idle\":\"2022-06-23T10:27:50.834629Z\",\"iopub.execute_input\":\"2022-06-23T10:27:47.70857Z\",\"shell.execute_reply\":\"2022-06-23T10:27:50.833567Z\"}" trusted="true"}
``` {.python}
import itertools
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report

metrics = ['accuracy',
                tf.keras.metrics.AUC(),
                tfa.metrics.CohenKappa(num_classes = 3),
                tfa.metrics.F1Score(num_classes = 3),
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall()]

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
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:27:50.836099Z\",\"shell.execute_reply.started\":\"2022-06-23T10:27:50.836446Z\",\"iopub.status.idle\":\"2022-06-23T10:27:50.84477Z\",\"iopub.execute_input\":\"2022-06-23T10:27:50.836485Z\",\"shell.execute_reply\":\"2022-06-23T10:27:50.843529Z\"}" trusted="true"}
``` {.python}
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
    print("ROC AUC score:", plot_roc_curves(y_true, y_pred, 3, class_labels))
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:27:50.846299Z\",\"shell.execute_reply.started\":\"2022-06-23T10:27:50.846912Z\",\"iopub.status.idle\":\"2022-06-23T10:27:52.187052Z\",\"iopub.execute_input\":\"2022-06-23T10:27:50.846955Z\",\"shell.execute_reply\":\"2022-06-23T10:27:52.185892Z\"}" trusted="true"}
``` {.python}
vgg_model = generate_model('vgg16', 3)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:27:52.188808Z\",\"shell.execute_reply.started\":\"2022-06-23T10:27:52.189418Z\",\"iopub.status.idle\":\"2022-06-23T10:35:32.923304Z\",\"iopub.execute_input\":\"2022-06-23T10:27:52.18946Z\",\"shell.execute_reply\":\"2022-06-23T10:35:32.922037Z\"}" trusted="true"}
``` {.python}
vgg_model, vgg_history = train_model(vgg_model, train_generator, val_generator, 50, tf.keras.optimizers.SGD(lr=0.001, momentum=0.9), metrics)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:35:32.926935Z\",\"shell.execute_reply.started\":\"2022-06-23T10:35:32.927344Z\",\"iopub.status.idle\":\"2022-06-23T10:35:40.937954Z\",\"iopub.execute_input\":\"2022-06-23T10:35:32.92738Z\",\"shell.execute_reply\":\"2022-06-23T10:35:40.936776Z\"}" trusted="true"}
``` {.python}
evaluate_model(vgg_model, vgg_history, test_generator)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:35:40.939813Z\",\"shell.execute_reply.started\":\"2022-06-23T10:35:40.940168Z\",\"iopub.status.idle\":\"2022-06-23T10:35:43.053054Z\",\"iopub.execute_input\":\"2022-06-23T10:35:40.940208Z\",\"shell.execute_reply\":\"2022-06-23T10:35:43.05193Z\"}" trusted="true"}
``` {.python}
vgg_model.save("/kaggle/working/vgg_model_weights.h5")
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:35:43.054584Z\",\"shell.execute_reply.started\":\"2022-06-23T10:35:43.054943Z\",\"iopub.status.idle\":\"2022-06-23T10:35:43.061595Z\",\"iopub.execute_input\":\"2022-06-23T10:35:43.054981Z\",\"shell.execute_reply\":\"2022-06-23T10:35:43.060378Z\"}" trusted="true"}
``` {.python}
import pandas as pd
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:35:43.063666Z\",\"shell.execute_reply.started\":\"2022-06-23T10:35:43.064438Z\",\"iopub.status.idle\":\"2022-06-23T10:35:43.080378Z\",\"iopub.execute_input\":\"2022-06-23T10:35:43.06448Z\",\"shell.execute_reply\":\"2022-06-23T10:35:43.079283Z\"}" trusted="true"}
``` {.python}
from keras.preprocessing.image import load_img,img_to_array
image_path= df_test['image_path'][101]
img = load_img(image_path, target_size=(224,224,3)) # stores image in PIL format
image_array=img_to_array(img)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:35:43.082062Z\",\"shell.execute_reply.started\":\"2022-06-23T10:35:43.083066Z\",\"iopub.status.idle\":\"2022-06-23T10:35:43.14128Z\",\"iopub.execute_input\":\"2022-06-23T10:35:43.08311Z\",\"shell.execute_reply\":\"2022-06-23T10:35:43.138235Z\"}" trusted="true"}
``` {.python}
from PIL import Image
display(Image.open(image_path))
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:35:43.142559Z\",\"shell.execute_reply.started\":\"2022-06-23T10:35:43.142882Z\",\"iopub.status.idle\":\"2022-06-23T10:35:43.154703Z\",\"iopub.execute_input\":\"2022-06-23T10:35:43.142916Z\",\"shell.execute_reply\":\"2022-06-23T10:35:43.152309Z\"}" trusted="true"}
``` {.python}
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
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:35:43.1613Z\",\"shell.execute_reply.started\":\"2022-06-23T10:35:43.161701Z\",\"iopub.status.idle\":\"2022-06-23T10:35:43.967314Z\",\"iopub.execute_input\":\"2022-06-23T10:35:43.16173Z\",\"shell.execute_reply\":\"2022-06-23T10:35:43.965964Z\"}" trusted="true"}
``` {.python}
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
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:35:43.969587Z\",\"shell.execute_reply.started\":\"2022-06-23T10:35:43.970232Z\",\"iopub.status.idle\":\"2022-06-23T10:35:43.98286Z\",\"iopub.execute_input\":\"2022-06-23T10:35:43.97028Z\",\"shell.execute_reply\":\"2022-06-23T10:35:43.981328Z\"}" trusted="true"}
``` {.python}
import matplotlib.cm as cm
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
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:35:43.989403Z\",\"shell.execute_reply.started\":\"2022-06-23T10:35:43.992108Z\",\"iopub.status.idle\":\"2022-06-23T10:35:44.131114Z\",\"iopub.execute_input\":\"2022-06-23T10:35:43.992178Z\",\"shell.execute_reply\":\"2022-06-23T10:35:44.129666Z\"}" trusted="true"}
``` {.python}
save_and_display_gradcam(image_path, heatmap,cam_path="/kaggle/working/GradCamTest.jpg")
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:35:44.132799Z\",\"shell.execute_reply.started\":\"2022-06-23T10:35:44.133448Z\",\"iopub.status.idle\":\"2022-06-23T10:35:56.66447Z\",\"iopub.execute_input\":\"2022-06-23T10:35:44.133491Z\",\"shell.execute_reply\":\"2022-06-23T10:35:56.663247Z\"}" trusted="true"}
``` {.python}
incres_model = generate_model('inceptionresnet', 3)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:35:56.665916Z\",\"shell.execute_reply.started\":\"2022-06-23T10:35:56.666303Z\",\"iopub.status.idle\":\"2022-06-23T10:46:39.685636Z\",\"iopub.execute_input\":\"2022-06-23T10:35:56.666342Z\",\"shell.execute_reply\":\"2022-06-23T10:46:39.684379Z\"}" trusted="true"}
``` {.python}
incres_model, incresincres_history = train_model(incres_model, train_generator, val_generator, 50, tf.keras.optimizers.SGD(lr=0.001, momentum=0.9), metrics)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:46:39.687523Z\",\"shell.execute_reply.started\":\"2022-06-23T10:46:39.687949Z\",\"iopub.status.idle\":\"2022-06-23T10:46:51.105984Z\",\"iopub.execute_input\":\"2022-06-23T10:46:39.687993Z\",\"shell.execute_reply\":\"2022-06-23T10:46:51.104742Z\"}" trusted="true"}
``` {.python}
evaluate_model(incres_model, incresincres_history, test_generator)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:46:51.107573Z\",\"shell.execute_reply.started\":\"2022-06-23T10:46:51.107945Z\",\"iopub.status.idle\":\"2022-06-23T10:46:51.122546Z\",\"iopub.execute_input\":\"2022-06-23T10:46:51.107987Z\",\"shell.execute_reply\":\"2022-06-23T10:46:51.12117Z\"}" trusted="true"}
``` {.python}
from keras.layers import Input, Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D,AveragePooling2D, BatchNormalization, PReLU, ReLU, SeparableConv2D
from keras.models import Model, Sequential
kernel_size = (3,3)
pool_size= (2,2)
first_filters = 64
second_filters = 128
third_filters = 256

dropout_conv = 0.3
dropout_dense = 0.3

# create a model with separable convolutional layers
def custom_model():
    inputs = Input((224, 224, 3))
    x = Conv2D(first_filters, kernel_size, activation = 'relu')(inputs)
    x = Conv2D(first_filters, kernel_size, activation = 'relu')(x)
    x = Conv2D(first_filters, kernel_size, activation = 'relu')(x)
    x = MaxPooling2D(pool_size = pool_size)(x)
    x = Dropout(dropout_conv)(x)
    x = Conv2D(second_filters, kernel_size, activation ='relu')(x)
    x = Conv2D(second_filters, kernel_size, activation ='relu')(x)
    x = Conv2D(second_filters, kernel_size, activation ='relu')(x)
    x = MaxPooling2D(pool_size = pool_size)(x)
    x = Dropout(dropout_conv)(x)
    x = Conv2D(third_filters, kernel_size, activation ='relu')(x)
    x = Conv2D(third_filters, kernel_size, activation ='relu')(x)
    x = Conv2D(third_filters, kernel_size, activation ='relu')(x)
    x = MaxPooling2D(pool_size = pool_size)(x)
    x = Dropout(dropout_conv)(x)
    x = Flatten()(x)
    x = Dense(4096)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(4096)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(3, activation = "softmax")(x)
    model = Model(inputs = inputs, outputs = x)
    model.summary
    return model
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:46:51.124091Z\",\"shell.execute_reply.started\":\"2022-06-23T10:46:51.124507Z\",\"iopub.status.idle\":\"2022-06-23T10:46:51.289105Z\",\"iopub.execute_input\":\"2022-06-23T10:46:51.12455Z\",\"shell.execute_reply\":\"2022-06-23T10:46:51.287945Z\"}" trusted="true"}
``` {.python}
model = custom_model()
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T10:46:51.290855Z\",\"shell.execute_reply.started\":\"2022-06-23T10:46:51.291304Z\",\"iopub.status.idle\":\"2022-06-23T11:21:20.800854Z\",\"iopub.execute_input\":\"2022-06-23T10:46:51.291348Z\",\"shell.execute_reply\":\"2022-06-23T11:21:20.79978Z\"}" trusted="true"}
``` {.python}
model, history = train_model(model, train_generator, val_generator, 50, tf.keras.optimizers.SGD(lr=0.001, momentum=0.9), metrics)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T11:21:20.803779Z\",\"shell.execute_reply.started\":\"2022-06-23T11:21:20.804127Z\",\"iopub.status.idle\":\"2022-06-23T11:21:27.268836Z\",\"iopub.execute_input\":\"2022-06-23T11:21:20.804162Z\",\"shell.execute_reply\":\"2022-06-23T11:21:27.267812Z\"}" trusted="true"}
``` {.python}
evaluate_model(model, history, test_generator)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T11:21:27.272214Z\",\"shell.execute_reply.started\":\"2022-06-23T11:21:27.272489Z\",\"iopub.status.idle\":\"2022-06-23T11:21:32.985924Z\",\"iopub.execute_input\":\"2022-06-23T11:21:27.272531Z\",\"shell.execute_reply\":\"2022-06-23T11:21:32.984848Z\"}" trusted="true"}
``` {.python}
inception_model = generate_model('inceptionv3', 3)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T11:21:32.987589Z\",\"shell.execute_reply.started\":\"2022-06-23T11:21:32.987984Z\",\"iopub.status.idle\":\"2022-06-23T11:26:18.852658Z\",\"iopub.execute_input\":\"2022-06-23T11:21:32.988023Z\",\"shell.execute_reply\":\"2022-06-23T11:26:18.851406Z\"}" trusted="true"}
``` {.python}
inception_model, inception_history = train_model(incres_model, train_generator, val_generator, 20, tf.keras.optimizers.SGD(lr=0.001, momentum=0.9), metrics)
```
:::

::: {.cell .code execution="{\"iopub.status.busy\":\"2022-06-23T11:26:18.854744Z\",\"shell.execute_reply.started\":\"2022-06-23T11:26:18.855192Z\",\"iopub.status.idle\":\"2022-06-23T11:26:29.80358Z\",\"iopub.execute_input\":\"2022-06-23T11:26:18.855238Z\",\"shell.execute_reply\":\"2022-06-23T11:26:29.802035Z\"}" trusted="true"}
``` {.python}
evaluate_model(inception_model, inception_history, test_generator)
```
:::
