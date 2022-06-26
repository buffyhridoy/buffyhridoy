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
```


```python
train_path = "../input/oct-dataset-duke-srinivasan-2014"
```


```python
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

    100%|██████████| 723/723 [00:00<00:00, 362346.97it/s]
    100%|██████████| 1407/1407 [00:00<00:00, 372162.81it/s]
    100%|██████████| 1101/1101 [00:00<00:00, 364103.82it/s]
    


```python
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)
```


```python
X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=0.5, random_state=42)
```


```python
sns.countplot(x = Y_train)
```




    <AxesSubplot:ylabel='count'>




![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_5_1.png)



```python
sns.countplot(x = Y_val)
```




    <AxesSubplot:ylabel='count'>




![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_6_1.png)



```python
sns.countplot(x = Y_test)
```




    <AxesSubplot:ylabel='count'>




![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_7_1.png)



```python
df_train = pd.DataFrame(list(zip(X_train, Y_train)), columns =['image_path', 'label'])
df_val = pd.DataFrame(list(zip(X_val, Y_val)), columns =['image_path', 'label'])
df_test = pd.DataFrame(list(zip(X_test, Y_test)), columns =['image_path', 'label'])
```


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

    Found 2261 validated image filenames belonging to 3 classes.
    Found 485 validated image filenames belonging to 3 classes.
    Found 485 validated image filenames belonging to 3 classes.
    


```python
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


```python
def train_model(model, train_generator, test_generator, num_epochs, optimizer, metrics):
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer, 
                  metrics=metrics)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=15, verbose=1)
    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5)
    print(model.summary())
    
    history = model.fit(train_generator, epochs=num_epochs, 
                        validation_data=test_generator, verbose=1,
                        callbacks = [early_stop, rlr])
    
    return model, history
```


```python
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
    print("ROC AUC score:", plot_roc_curves(y_true, y_pred, 3, class_labels))
```


```python
vgg_model = generate_model('vgg16', 3)
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58892288/58889256 [==============================] - 0s 0us/step
    


```python
vgg_model, vgg_history = train_model(vgg_model, train_generator, val_generator, 50, tf.keras.optimizers.SGD(lr=0.001, momentum=0.9), metrics)
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 25088)             0         
    _________________________________________________________________
    dense (Dense)                (None, 4096)              102764544 
    _________________________________________________________________
    re_lu (ReLU)                 (None, 4096)              0         
    _________________________________________________________________
    dropout (Dropout)            (None, 4096)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 4096)              16781312  
    _________________________________________________________________
    re_lu_1 (ReLU)               (None, 4096)              0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 4096)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 3)                 12291     
    =================================================================
    Total params: 134,272,835
    Trainable params: 124,277,763
    Non-trainable params: 9,995,072
    _________________________________________________________________
    None
    Epoch 1/50
    142/142 [==============================] - 62s 364ms/step - loss: 1.0227 - accuracy: 0.5732 - auc: 0.7437 - cohen_kappa: 0.3158 - f1_score: 0.5311 - precision: 0.6123 - recall: 0.5142 - val_loss: 0.1324 - val_accuracy: 0.9505 - val_auc: 0.9950 - val_cohen_kappa: 0.9240 - val_f1_score: 0.9500 - val_precision: 0.9563 - val_recall: 0.9464
    Epoch 2/50
    142/142 [==============================] - 34s 240ms/step - loss: 0.1575 - accuracy: 0.9387 - auc: 0.9929 - cohen_kappa: 0.9045 - f1_score: 0.9319 - precision: 0.9463 - recall: 0.9326 - val_loss: 0.0846 - val_accuracy: 0.9649 - val_auc: 0.9981 - val_cohen_kappa: 0.9464 - val_f1_score: 0.9632 - val_precision: 0.9649 - val_recall: 0.9649
    Epoch 3/50
    142/142 [==============================] - 34s 240ms/step - loss: 0.0875 - accuracy: 0.9711 - auc: 0.9979 - cohen_kappa: 0.9549 - f1_score: 0.9711 - precision: 0.9733 - recall: 0.9708 - val_loss: 0.1309 - val_accuracy: 0.9670 - val_auc: 0.9909 - val_cohen_kappa: 0.9493 - val_f1_score: 0.9695 - val_precision: 0.9670 - val_recall: 0.9670
    Epoch 4/50
    142/142 [==============================] - 34s 243ms/step - loss: 0.0576 - accuracy: 0.9788 - auc: 0.9982 - cohen_kappa: 0.9667 - f1_score: 0.9788 - precision: 0.9798 - recall: 0.9788 - val_loss: 0.0788 - val_accuracy: 0.9753 - val_auc: 0.9982 - val_cohen_kappa: 0.9620 - val_f1_score: 0.9780 - val_precision: 0.9753 - val_recall: 0.9753
    Epoch 5/50
    142/142 [==============================] - 35s 247ms/step - loss: 0.0242 - accuracy: 0.9931 - auc: 0.9997 - cohen_kappa: 0.9893 - f1_score: 0.9924 - precision: 0.9935 - recall: 0.9931 - val_loss: 0.0206 - val_accuracy: 0.9938 - val_auc: 0.9999 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9940 - val_precision: 0.9938 - val_recall: 0.9918
    Epoch 6/50
    142/142 [==============================] - 35s 249ms/step - loss: 0.0400 - accuracy: 0.9846 - auc: 0.9995 - cohen_kappa: 0.9762 - f1_score: 0.9837 - precision: 0.9862 - recall: 0.9846 - val_loss: 0.0160 - val_accuracy: 0.9979 - val_auc: 1.0000 - val_cohen_kappa: 0.9968 - val_f1_score: 0.9976 - val_precision: 0.9979 - val_recall: 0.9979
    Epoch 7/50
    142/142 [==============================] - 35s 247ms/step - loss: 0.0386 - accuracy: 0.9874 - auc: 0.9988 - cohen_kappa: 0.9804 - f1_score: 0.9879 - precision: 0.9896 - recall: 0.9874 - val_loss: 0.0106 - val_accuracy: 0.9979 - val_auc: 1.0000 - val_cohen_kappa: 0.9968 - val_f1_score: 0.9976 - val_precision: 0.9979 - val_recall: 0.9979
    Epoch 8/50
    142/142 [==============================] - 35s 245ms/step - loss: 0.0235 - accuracy: 0.9930 - auc: 0.9998 - cohen_kappa: 0.9891 - f1_score: 0.9936 - precision: 0.9930 - recall: 0.9930 - val_loss: 0.0829 - val_accuracy: 0.9711 - val_auc: 0.9977 - val_cohen_kappa: 0.9557 - val_f1_score: 0.9731 - val_precision: 0.9711 - val_recall: 0.9711
    Epoch 9/50
    142/142 [==============================] - 34s 243ms/step - loss: 0.0179 - accuracy: 0.9932 - auc: 0.9999 - cohen_kappa: 0.9893 - f1_score: 0.9928 - precision: 0.9932 - recall: 0.9932 - val_loss: 0.0108 - val_accuracy: 0.9959 - val_auc: 1.0000 - val_cohen_kappa: 0.9937 - val_f1_score: 0.9958 - val_precision: 0.9959 - val_recall: 0.9959
    Epoch 10/50
    142/142 [==============================] - 35s 248ms/step - loss: 0.0292 - accuracy: 0.9909 - auc: 0.9997 - cohen_kappa: 0.9858 - f1_score: 0.9915 - precision: 0.9914 - recall: 0.9909 - val_loss: 0.0097 - val_accuracy: 0.9979 - val_auc: 1.0000 - val_cohen_kappa: 0.9968 - val_f1_score: 0.9976 - val_precision: 0.9979 - val_recall: 0.9979
    Epoch 11/50
    142/142 [==============================] - 35s 248ms/step - loss: 0.0155 - accuracy: 0.9944 - auc: 0.9999 - cohen_kappa: 0.9912 - f1_score: 0.9940 - precision: 0.9944 - recall: 0.9944 - val_loss: 0.0452 - val_accuracy: 0.9856 - val_auc: 0.9971 - val_cohen_kappa: 0.9779 - val_f1_score: 0.9872 - val_precision: 0.9855 - val_recall: 0.9835
    Epoch 12/50
    142/142 [==============================] - 36s 253ms/step - loss: 0.0192 - accuracy: 0.9951 - auc: 0.9993 - cohen_kappa: 0.9924 - f1_score: 0.9948 - precision: 0.9958 - recall: 0.9951 - val_loss: 0.0052 - val_accuracy: 0.9979 - val_auc: 1.0000 - val_cohen_kappa: 0.9968 - val_f1_score: 0.9982 - val_precision: 0.9979 - val_recall: 0.9979
    Epoch 13/50
    142/142 [==============================] - 37s 260ms/step - loss: 0.0090 - accuracy: 0.9968 - auc: 1.0000 - cohen_kappa: 0.9950 - f1_score: 0.9972 - precision: 0.9968 - recall: 0.9968 - val_loss: 0.0198 - val_accuracy: 0.9938 - val_auc: 0.9999 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9945 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 14/50
    142/142 [==============================] - 36s 256ms/step - loss: 0.0031 - accuracy: 0.9999 - auc: 1.0000 - cohen_kappa: 0.9999 - f1_score: 0.9999 - precision: 0.9999 - recall: 0.9999 - val_loss: 0.0165 - val_accuracy: 0.9959 - val_auc: 0.9999 - val_cohen_kappa: 0.9937 - val_f1_score: 0.9964 - val_precision: 0.9959 - val_recall: 0.9959
    Epoch 15/50
    142/142 [==============================] - 37s 257ms/step - loss: 0.0108 - accuracy: 0.9956 - auc: 1.0000 - cohen_kappa: 0.9931 - f1_score: 0.9951 - precision: 0.9955 - recall: 0.9953 - val_loss: 0.0198 - val_accuracy: 0.9938 - val_auc: 0.9999 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9940 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 16/50
    142/142 [==============================] - 36s 250ms/step - loss: 0.0148 - accuracy: 0.9934 - auc: 0.9999 - cohen_kappa: 0.9897 - f1_score: 0.9919 - precision: 0.9938 - recall: 0.9934 - val_loss: 0.0213 - val_accuracy: 0.9959 - val_auc: 0.9998 - val_cohen_kappa: 0.9937 - val_f1_score: 0.9964 - val_precision: 0.9959 - val_recall: 0.9959
    Epoch 17/50
    142/142 [==============================] - 35s 246ms/step - loss: 0.0052 - accuracy: 0.9985 - auc: 1.0000 - cohen_kappa: 0.9977 - f1_score: 0.9984 - precision: 0.9985 - recall: 0.9985 - val_loss: 0.0142 - val_accuracy: 0.9938 - val_auc: 1.0000 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9939 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 18/50
    142/142 [==============================] - 35s 247ms/step - loss: 0.0145 - accuracy: 0.9940 - auc: 0.9999 - cohen_kappa: 0.9906 - f1_score: 0.9936 - precision: 0.9940 - recall: 0.9940 - val_loss: 0.0169 - val_accuracy: 0.9959 - val_auc: 0.9999 - val_cohen_kappa: 0.9937 - val_f1_score: 0.9964 - val_precision: 0.9959 - val_recall: 0.9959
    Epoch 19/50
    142/142 [==============================] - 35s 246ms/step - loss: 0.0050 - accuracy: 0.9980 - auc: 1.0000 - cohen_kappa: 0.9968 - f1_score: 0.9982 - precision: 0.9980 - recall: 0.9980 - val_loss: 0.0205 - val_accuracy: 0.9959 - val_auc: 0.9999 - val_cohen_kappa: 0.9937 - val_f1_score: 0.9964 - val_precision: 0.9959 - val_recall: 0.9938
    Epoch 20/50
    142/142 [==============================] - 35s 246ms/step - loss: 0.0077 - accuracy: 0.9963 - auc: 1.0000 - cohen_kappa: 0.9941 - f1_score: 0.9966 - precision: 0.9963 - recall: 0.9963 - val_loss: 0.0185 - val_accuracy: 0.9959 - val_auc: 0.9999 - val_cohen_kappa: 0.9937 - val_f1_score: 0.9964 - val_precision: 0.9959 - val_recall: 0.9959
    Epoch 21/50
    142/142 [==============================] - 35s 248ms/step - loss: 0.0022 - accuracy: 0.9999 - auc: 1.0000 - cohen_kappa: 0.9999 - f1_score: 0.9999 - precision: 0.9999 - recall: 0.9999 - val_loss: 0.0182 - val_accuracy: 0.9959 - val_auc: 0.9999 - val_cohen_kappa: 0.9937 - val_f1_score: 0.9964 - val_precision: 0.9959 - val_recall: 0.9959
    Epoch 00021: early stopping
    


```python
evaluate_model(vgg_model, vgg_history, test_generator)
```

    
    Test set accuracy: 1.0 
    
    31/31 [==============================] - 3s 72ms/step
    
                  precision    recall  f1-score   support
    
             AMD       1.00      1.00      1.00       121
             DME       1.00      1.00      1.00       155
          NORMAL       1.00      1.00      1.00       209
    
        accuracy                           1.00       485
       macro avg       1.00      1.00      1.00       485
    weighted avg       1.00      1.00      1.00       485
    
    


![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_16_1.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_16_2.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_16_3.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_16_4.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_16_5.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_16_6.png)


    ROC AUC score: 1.0
    


```python
vgg_model.save("/kaggle/working/vgg_model_weights.h5")
```


```python
import pandas as pd
```


```python
from keras.preprocessing.image import load_img,img_to_array
image_path= df_test['image_path'][101]
img = load_img(image_path, target_size=(224,224,3)) # stores image in PIL format
image_array=img_to_array(img)

```


```python
from PIL import Image
display(Image.open(image_path))
```


![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_20_0.png)



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


![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_22_0.png)



```python
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


```python
save_and_display_gradcam(image_path, heatmap,cam_path="/kaggle/working/GradCamTest.jpg")
```


![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_24_0.png)



```python
incres_model = generate_model('inceptionresnet', 3)

```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/inception_resnet_v2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5
    219062272/219055592 [==============================] - 6s 0us/step
    


```python
incres_model, incres_history = train_model(incres_model, train_generator, val_generator, 50, tf.keras.optimizers.SGD(lr=0.001, momentum=0.9), metrics)
```

    Model: "model_2"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_2 (InputLayer)            [(None, 224, 224, 3) 0                                            
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, 111, 111, 32) 864         input_2[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization (BatchNorma (None, 111, 111, 32) 96          conv2d[0][0]                     
    __________________________________________________________________________________________________
    activation (Activation)         (None, 111, 111, 32) 0           batch_normalization[0][0]        
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 109, 109, 32) 9216        activation[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, 109, 109, 32) 96          conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    activation_1 (Activation)       (None, 109, 109, 32) 0           batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 109, 109, 64) 18432       activation_1[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, 109, 109, 64) 192         conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    activation_2 (Activation)       (None, 109, 109, 64) 0           batch_normalization_2[0][0]      
    __________________________________________________________________________________________________
    max_pooling2d (MaxPooling2D)    (None, 54, 54, 64)   0           activation_2[0][0]               
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, 54, 54, 80)   5120        max_pooling2d[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_3 (BatchNor (None, 54, 54, 80)   240         conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    activation_3 (Activation)       (None, 54, 54, 80)   0           batch_normalization_3[0][0]      
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, 52, 52, 192)  138240      activation_3[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_4 (BatchNor (None, 52, 52, 192)  576         conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    activation_4 (Activation)       (None, 52, 52, 192)  0           batch_normalization_4[0][0]      
    __________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)  (None, 25, 25, 192)  0           activation_4[0][0]               
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, 25, 25, 64)   12288       max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_8 (BatchNor (None, 25, 25, 64)   192         conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    activation_8 (Activation)       (None, 25, 25, 64)   0           batch_normalization_8[0][0]      
    __________________________________________________________________________________________________
    conv2d_6 (Conv2D)               (None, 25, 25, 48)   9216        max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    conv2d_9 (Conv2D)               (None, 25, 25, 96)   55296       activation_8[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_6 (BatchNor (None, 25, 25, 48)   144         conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_9 (BatchNor (None, 25, 25, 96)   288         conv2d_9[0][0]                   
    __________________________________________________________________________________________________
    activation_6 (Activation)       (None, 25, 25, 48)   0           batch_normalization_6[0][0]      
    __________________________________________________________________________________________________
    activation_9 (Activation)       (None, 25, 25, 96)   0           batch_normalization_9[0][0]      
    __________________________________________________________________________________________________
    average_pooling2d (AveragePooli (None, 25, 25, 192)  0           max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, 25, 25, 96)   18432       max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    conv2d_7 (Conv2D)               (None, 25, 25, 64)   76800       activation_6[0][0]               
    __________________________________________________________________________________________________
    conv2d_10 (Conv2D)              (None, 25, 25, 96)   82944       activation_9[0][0]               
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, 25, 25, 64)   12288       average_pooling2d[0][0]          
    __________________________________________________________________________________________________
    batch_normalization_5 (BatchNor (None, 25, 25, 96)   288         conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_7 (BatchNor (None, 25, 25, 64)   192         conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_10 (BatchNo (None, 25, 25, 96)   288         conv2d_10[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_11 (BatchNo (None, 25, 25, 64)   192         conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    activation_5 (Activation)       (None, 25, 25, 96)   0           batch_normalization_5[0][0]      
    __________________________________________________________________________________________________
    activation_7 (Activation)       (None, 25, 25, 64)   0           batch_normalization_7[0][0]      
    __________________________________________________________________________________________________
    activation_10 (Activation)      (None, 25, 25, 96)   0           batch_normalization_10[0][0]     
    __________________________________________________________________________________________________
    activation_11 (Activation)      (None, 25, 25, 64)   0           batch_normalization_11[0][0]     
    __________________________________________________________________________________________________
    mixed_5b (Concatenate)          (None, 25, 25, 320)  0           activation_5[0][0]               
                                                                     activation_7[0][0]               
                                                                     activation_10[0][0]              
                                                                     activation_11[0][0]              
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, 25, 25, 32)   10240       mixed_5b[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_15 (BatchNo (None, 25, 25, 32)   96          conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    activation_15 (Activation)      (None, 25, 25, 32)   0           batch_normalization_15[0][0]     
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, 25, 25, 32)   10240       mixed_5b[0][0]                   
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, 25, 25, 48)   13824       activation_15[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_13 (BatchNo (None, 25, 25, 32)   96          conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_16 (BatchNo (None, 25, 25, 48)   144         conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    activation_13 (Activation)      (None, 25, 25, 32)   0           batch_normalization_13[0][0]     
    __________________________________________________________________________________________________
    activation_16 (Activation)      (None, 25, 25, 48)   0           batch_normalization_16[0][0]     
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, 25, 25, 32)   10240       mixed_5b[0][0]                   
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, 25, 25, 32)   9216        activation_13[0][0]              
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, 25, 25, 64)   27648       activation_16[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_12 (BatchNo (None, 25, 25, 32)   96          conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_14 (BatchNo (None, 25, 25, 32)   96          conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_17 (BatchNo (None, 25, 25, 64)   192         conv2d_17[0][0]                  
    __________________________________________________________________________________________________
    activation_12 (Activation)      (None, 25, 25, 32)   0           batch_normalization_12[0][0]     
    __________________________________________________________________________________________________
    activation_14 (Activation)      (None, 25, 25, 32)   0           batch_normalization_14[0][0]     
    __________________________________________________________________________________________________
    activation_17 (Activation)      (None, 25, 25, 64)   0           batch_normalization_17[0][0]     
    __________________________________________________________________________________________________
    block35_1_mixed (Concatenate)   (None, 25, 25, 128)  0           activation_12[0][0]              
                                                                     activation_14[0][0]              
                                                                     activation_17[0][0]              
    __________________________________________________________________________________________________
    block35_1_conv (Conv2D)         (None, 25, 25, 320)  41280       block35_1_mixed[0][0]            
    __________________________________________________________________________________________________
    block35_1 (Lambda)              (None, 25, 25, 320)  0           mixed_5b[0][0]                   
                                                                     block35_1_conv[0][0]             
    __________________________________________________________________________________________________
    block35_1_ac (Activation)       (None, 25, 25, 320)  0           block35_1[0][0]                  
    __________________________________________________________________________________________________
    conv2d_21 (Conv2D)              (None, 25, 25, 32)   10240       block35_1_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_21 (BatchNo (None, 25, 25, 32)   96          conv2d_21[0][0]                  
    __________________________________________________________________________________________________
    activation_21 (Activation)      (None, 25, 25, 32)   0           batch_normalization_21[0][0]     
    __________________________________________________________________________________________________
    conv2d_19 (Conv2D)              (None, 25, 25, 32)   10240       block35_1_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_22 (Conv2D)              (None, 25, 25, 48)   13824       activation_21[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_19 (BatchNo (None, 25, 25, 32)   96          conv2d_19[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_22 (BatchNo (None, 25, 25, 48)   144         conv2d_22[0][0]                  
    __________________________________________________________________________________________________
    activation_19 (Activation)      (None, 25, 25, 32)   0           batch_normalization_19[0][0]     
    __________________________________________________________________________________________________
    activation_22 (Activation)      (None, 25, 25, 48)   0           batch_normalization_22[0][0]     
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, 25, 25, 32)   10240       block35_1_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_20 (Conv2D)              (None, 25, 25, 32)   9216        activation_19[0][0]              
    __________________________________________________________________________________________________
    conv2d_23 (Conv2D)              (None, 25, 25, 64)   27648       activation_22[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_18 (BatchNo (None, 25, 25, 32)   96          conv2d_18[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_20 (BatchNo (None, 25, 25, 32)   96          conv2d_20[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_23 (BatchNo (None, 25, 25, 64)   192         conv2d_23[0][0]                  
    __________________________________________________________________________________________________
    activation_18 (Activation)      (None, 25, 25, 32)   0           batch_normalization_18[0][0]     
    __________________________________________________________________________________________________
    activation_20 (Activation)      (None, 25, 25, 32)   0           batch_normalization_20[0][0]     
    __________________________________________________________________________________________________
    activation_23 (Activation)      (None, 25, 25, 64)   0           batch_normalization_23[0][0]     
    __________________________________________________________________________________________________
    block35_2_mixed (Concatenate)   (None, 25, 25, 128)  0           activation_18[0][0]              
                                                                     activation_20[0][0]              
                                                                     activation_23[0][0]              
    __________________________________________________________________________________________________
    block35_2_conv (Conv2D)         (None, 25, 25, 320)  41280       block35_2_mixed[0][0]            
    __________________________________________________________________________________________________
    block35_2 (Lambda)              (None, 25, 25, 320)  0           block35_1_ac[0][0]               
                                                                     block35_2_conv[0][0]             
    __________________________________________________________________________________________________
    block35_2_ac (Activation)       (None, 25, 25, 320)  0           block35_2[0][0]                  
    __________________________________________________________________________________________________
    conv2d_27 (Conv2D)              (None, 25, 25, 32)   10240       block35_2_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_27 (BatchNo (None, 25, 25, 32)   96          conv2d_27[0][0]                  
    __________________________________________________________________________________________________
    activation_27 (Activation)      (None, 25, 25, 32)   0           batch_normalization_27[0][0]     
    __________________________________________________________________________________________________
    conv2d_25 (Conv2D)              (None, 25, 25, 32)   10240       block35_2_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_28 (Conv2D)              (None, 25, 25, 48)   13824       activation_27[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_25 (BatchNo (None, 25, 25, 32)   96          conv2d_25[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_28 (BatchNo (None, 25, 25, 48)   144         conv2d_28[0][0]                  
    __________________________________________________________________________________________________
    activation_25 (Activation)      (None, 25, 25, 32)   0           batch_normalization_25[0][0]     
    __________________________________________________________________________________________________
    activation_28 (Activation)      (None, 25, 25, 48)   0           batch_normalization_28[0][0]     
    __________________________________________________________________________________________________
    conv2d_24 (Conv2D)              (None, 25, 25, 32)   10240       block35_2_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_26 (Conv2D)              (None, 25, 25, 32)   9216        activation_25[0][0]              
    __________________________________________________________________________________________________
    conv2d_29 (Conv2D)              (None, 25, 25, 64)   27648       activation_28[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_24 (BatchNo (None, 25, 25, 32)   96          conv2d_24[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_26 (BatchNo (None, 25, 25, 32)   96          conv2d_26[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_29 (BatchNo (None, 25, 25, 64)   192         conv2d_29[0][0]                  
    __________________________________________________________________________________________________
    activation_24 (Activation)      (None, 25, 25, 32)   0           batch_normalization_24[0][0]     
    __________________________________________________________________________________________________
    activation_26 (Activation)      (None, 25, 25, 32)   0           batch_normalization_26[0][0]     
    __________________________________________________________________________________________________
    activation_29 (Activation)      (None, 25, 25, 64)   0           batch_normalization_29[0][0]     
    __________________________________________________________________________________________________
    block35_3_mixed (Concatenate)   (None, 25, 25, 128)  0           activation_24[0][0]              
                                                                     activation_26[0][0]              
                                                                     activation_29[0][0]              
    __________________________________________________________________________________________________
    block35_3_conv (Conv2D)         (None, 25, 25, 320)  41280       block35_3_mixed[0][0]            
    __________________________________________________________________________________________________
    block35_3 (Lambda)              (None, 25, 25, 320)  0           block35_2_ac[0][0]               
                                                                     block35_3_conv[0][0]             
    __________________________________________________________________________________________________
    block35_3_ac (Activation)       (None, 25, 25, 320)  0           block35_3[0][0]                  
    __________________________________________________________________________________________________
    conv2d_33 (Conv2D)              (None, 25, 25, 32)   10240       block35_3_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_33 (BatchNo (None, 25, 25, 32)   96          conv2d_33[0][0]                  
    __________________________________________________________________________________________________
    activation_33 (Activation)      (None, 25, 25, 32)   0           batch_normalization_33[0][0]     
    __________________________________________________________________________________________________
    conv2d_31 (Conv2D)              (None, 25, 25, 32)   10240       block35_3_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_34 (Conv2D)              (None, 25, 25, 48)   13824       activation_33[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_31 (BatchNo (None, 25, 25, 32)   96          conv2d_31[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_34 (BatchNo (None, 25, 25, 48)   144         conv2d_34[0][0]                  
    __________________________________________________________________________________________________
    activation_31 (Activation)      (None, 25, 25, 32)   0           batch_normalization_31[0][0]     
    __________________________________________________________________________________________________
    activation_34 (Activation)      (None, 25, 25, 48)   0           batch_normalization_34[0][0]     
    __________________________________________________________________________________________________
    conv2d_30 (Conv2D)              (None, 25, 25, 32)   10240       block35_3_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_32 (Conv2D)              (None, 25, 25, 32)   9216        activation_31[0][0]              
    __________________________________________________________________________________________________
    conv2d_35 (Conv2D)              (None, 25, 25, 64)   27648       activation_34[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_30 (BatchNo (None, 25, 25, 32)   96          conv2d_30[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_32 (BatchNo (None, 25, 25, 32)   96          conv2d_32[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_35 (BatchNo (None, 25, 25, 64)   192         conv2d_35[0][0]                  
    __________________________________________________________________________________________________
    activation_30 (Activation)      (None, 25, 25, 32)   0           batch_normalization_30[0][0]     
    __________________________________________________________________________________________________
    activation_32 (Activation)      (None, 25, 25, 32)   0           batch_normalization_32[0][0]     
    __________________________________________________________________________________________________
    activation_35 (Activation)      (None, 25, 25, 64)   0           batch_normalization_35[0][0]     
    __________________________________________________________________________________________________
    block35_4_mixed (Concatenate)   (None, 25, 25, 128)  0           activation_30[0][0]              
                                                                     activation_32[0][0]              
                                                                     activation_35[0][0]              
    __________________________________________________________________________________________________
    block35_4_conv (Conv2D)         (None, 25, 25, 320)  41280       block35_4_mixed[0][0]            
    __________________________________________________________________________________________________
    block35_4 (Lambda)              (None, 25, 25, 320)  0           block35_3_ac[0][0]               
                                                                     block35_4_conv[0][0]             
    __________________________________________________________________________________________________
    block35_4_ac (Activation)       (None, 25, 25, 320)  0           block35_4[0][0]                  
    __________________________________________________________________________________________________
    conv2d_39 (Conv2D)              (None, 25, 25, 32)   10240       block35_4_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_39 (BatchNo (None, 25, 25, 32)   96          conv2d_39[0][0]                  
    __________________________________________________________________________________________________
    activation_39 (Activation)      (None, 25, 25, 32)   0           batch_normalization_39[0][0]     
    __________________________________________________________________________________________________
    conv2d_37 (Conv2D)              (None, 25, 25, 32)   10240       block35_4_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_40 (Conv2D)              (None, 25, 25, 48)   13824       activation_39[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_37 (BatchNo (None, 25, 25, 32)   96          conv2d_37[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_40 (BatchNo (None, 25, 25, 48)   144         conv2d_40[0][0]                  
    __________________________________________________________________________________________________
    activation_37 (Activation)      (None, 25, 25, 32)   0           batch_normalization_37[0][0]     
    __________________________________________________________________________________________________
    activation_40 (Activation)      (None, 25, 25, 48)   0           batch_normalization_40[0][0]     
    __________________________________________________________________________________________________
    conv2d_36 (Conv2D)              (None, 25, 25, 32)   10240       block35_4_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_38 (Conv2D)              (None, 25, 25, 32)   9216        activation_37[0][0]              
    __________________________________________________________________________________________________
    conv2d_41 (Conv2D)              (None, 25, 25, 64)   27648       activation_40[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_36 (BatchNo (None, 25, 25, 32)   96          conv2d_36[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_38 (BatchNo (None, 25, 25, 32)   96          conv2d_38[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_41 (BatchNo (None, 25, 25, 64)   192         conv2d_41[0][0]                  
    __________________________________________________________________________________________________
    activation_36 (Activation)      (None, 25, 25, 32)   0           batch_normalization_36[0][0]     
    __________________________________________________________________________________________________
    activation_38 (Activation)      (None, 25, 25, 32)   0           batch_normalization_38[0][0]     
    __________________________________________________________________________________________________
    activation_41 (Activation)      (None, 25, 25, 64)   0           batch_normalization_41[0][0]     
    __________________________________________________________________________________________________
    block35_5_mixed (Concatenate)   (None, 25, 25, 128)  0           activation_36[0][0]              
                                                                     activation_38[0][0]              
                                                                     activation_41[0][0]              
    __________________________________________________________________________________________________
    block35_5_conv (Conv2D)         (None, 25, 25, 320)  41280       block35_5_mixed[0][0]            
    __________________________________________________________________________________________________
    block35_5 (Lambda)              (None, 25, 25, 320)  0           block35_4_ac[0][0]               
                                                                     block35_5_conv[0][0]             
    __________________________________________________________________________________________________
    block35_5_ac (Activation)       (None, 25, 25, 320)  0           block35_5[0][0]                  
    __________________________________________________________________________________________________
    conv2d_45 (Conv2D)              (None, 25, 25, 32)   10240       block35_5_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_45 (BatchNo (None, 25, 25, 32)   96          conv2d_45[0][0]                  
    __________________________________________________________________________________________________
    activation_45 (Activation)      (None, 25, 25, 32)   0           batch_normalization_45[0][0]     
    __________________________________________________________________________________________________
    conv2d_43 (Conv2D)              (None, 25, 25, 32)   10240       block35_5_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_46 (Conv2D)              (None, 25, 25, 48)   13824       activation_45[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_43 (BatchNo (None, 25, 25, 32)   96          conv2d_43[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_46 (BatchNo (None, 25, 25, 48)   144         conv2d_46[0][0]                  
    __________________________________________________________________________________________________
    activation_43 (Activation)      (None, 25, 25, 32)   0           batch_normalization_43[0][0]     
    __________________________________________________________________________________________________
    activation_46 (Activation)      (None, 25, 25, 48)   0           batch_normalization_46[0][0]     
    __________________________________________________________________________________________________
    conv2d_42 (Conv2D)              (None, 25, 25, 32)   10240       block35_5_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_44 (Conv2D)              (None, 25, 25, 32)   9216        activation_43[0][0]              
    __________________________________________________________________________________________________
    conv2d_47 (Conv2D)              (None, 25, 25, 64)   27648       activation_46[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_42 (BatchNo (None, 25, 25, 32)   96          conv2d_42[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_44 (BatchNo (None, 25, 25, 32)   96          conv2d_44[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_47 (BatchNo (None, 25, 25, 64)   192         conv2d_47[0][0]                  
    __________________________________________________________________________________________________
    activation_42 (Activation)      (None, 25, 25, 32)   0           batch_normalization_42[0][0]     
    __________________________________________________________________________________________________
    activation_44 (Activation)      (None, 25, 25, 32)   0           batch_normalization_44[0][0]     
    __________________________________________________________________________________________________
    activation_47 (Activation)      (None, 25, 25, 64)   0           batch_normalization_47[0][0]     
    __________________________________________________________________________________________________
    block35_6_mixed (Concatenate)   (None, 25, 25, 128)  0           activation_42[0][0]              
                                                                     activation_44[0][0]              
                                                                     activation_47[0][0]              
    __________________________________________________________________________________________________
    block35_6_conv (Conv2D)         (None, 25, 25, 320)  41280       block35_6_mixed[0][0]            
    __________________________________________________________________________________________________
    block35_6 (Lambda)              (None, 25, 25, 320)  0           block35_5_ac[0][0]               
                                                                     block35_6_conv[0][0]             
    __________________________________________________________________________________________________
    block35_6_ac (Activation)       (None, 25, 25, 320)  0           block35_6[0][0]                  
    __________________________________________________________________________________________________
    conv2d_51 (Conv2D)              (None, 25, 25, 32)   10240       block35_6_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_51 (BatchNo (None, 25, 25, 32)   96          conv2d_51[0][0]                  
    __________________________________________________________________________________________________
    activation_51 (Activation)      (None, 25, 25, 32)   0           batch_normalization_51[0][0]     
    __________________________________________________________________________________________________
    conv2d_49 (Conv2D)              (None, 25, 25, 32)   10240       block35_6_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_52 (Conv2D)              (None, 25, 25, 48)   13824       activation_51[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_49 (BatchNo (None, 25, 25, 32)   96          conv2d_49[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_52 (BatchNo (None, 25, 25, 48)   144         conv2d_52[0][0]                  
    __________________________________________________________________________________________________
    activation_49 (Activation)      (None, 25, 25, 32)   0           batch_normalization_49[0][0]     
    __________________________________________________________________________________________________
    activation_52 (Activation)      (None, 25, 25, 48)   0           batch_normalization_52[0][0]     
    __________________________________________________________________________________________________
    conv2d_48 (Conv2D)              (None, 25, 25, 32)   10240       block35_6_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_50 (Conv2D)              (None, 25, 25, 32)   9216        activation_49[0][0]              
    __________________________________________________________________________________________________
    conv2d_53 (Conv2D)              (None, 25, 25, 64)   27648       activation_52[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_48 (BatchNo (None, 25, 25, 32)   96          conv2d_48[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_50 (BatchNo (None, 25, 25, 32)   96          conv2d_50[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_53 (BatchNo (None, 25, 25, 64)   192         conv2d_53[0][0]                  
    __________________________________________________________________________________________________
    activation_48 (Activation)      (None, 25, 25, 32)   0           batch_normalization_48[0][0]     
    __________________________________________________________________________________________________
    activation_50 (Activation)      (None, 25, 25, 32)   0           batch_normalization_50[0][0]     
    __________________________________________________________________________________________________
    activation_53 (Activation)      (None, 25, 25, 64)   0           batch_normalization_53[0][0]     
    __________________________________________________________________________________________________
    block35_7_mixed (Concatenate)   (None, 25, 25, 128)  0           activation_48[0][0]              
                                                                     activation_50[0][0]              
                                                                     activation_53[0][0]              
    __________________________________________________________________________________________________
    block35_7_conv (Conv2D)         (None, 25, 25, 320)  41280       block35_7_mixed[0][0]            
    __________________________________________________________________________________________________
    block35_7 (Lambda)              (None, 25, 25, 320)  0           block35_6_ac[0][0]               
                                                                     block35_7_conv[0][0]             
    __________________________________________________________________________________________________
    block35_7_ac (Activation)       (None, 25, 25, 320)  0           block35_7[0][0]                  
    __________________________________________________________________________________________________
    conv2d_57 (Conv2D)              (None, 25, 25, 32)   10240       block35_7_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_57 (BatchNo (None, 25, 25, 32)   96          conv2d_57[0][0]                  
    __________________________________________________________________________________________________
    activation_57 (Activation)      (None, 25, 25, 32)   0           batch_normalization_57[0][0]     
    __________________________________________________________________________________________________
    conv2d_55 (Conv2D)              (None, 25, 25, 32)   10240       block35_7_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_58 (Conv2D)              (None, 25, 25, 48)   13824       activation_57[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_55 (BatchNo (None, 25, 25, 32)   96          conv2d_55[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_58 (BatchNo (None, 25, 25, 48)   144         conv2d_58[0][0]                  
    __________________________________________________________________________________________________
    activation_55 (Activation)      (None, 25, 25, 32)   0           batch_normalization_55[0][0]     
    __________________________________________________________________________________________________
    activation_58 (Activation)      (None, 25, 25, 48)   0           batch_normalization_58[0][0]     
    __________________________________________________________________________________________________
    conv2d_54 (Conv2D)              (None, 25, 25, 32)   10240       block35_7_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_56 (Conv2D)              (None, 25, 25, 32)   9216        activation_55[0][0]              
    __________________________________________________________________________________________________
    conv2d_59 (Conv2D)              (None, 25, 25, 64)   27648       activation_58[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_54 (BatchNo (None, 25, 25, 32)   96          conv2d_54[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_56 (BatchNo (None, 25, 25, 32)   96          conv2d_56[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_59 (BatchNo (None, 25, 25, 64)   192         conv2d_59[0][0]                  
    __________________________________________________________________________________________________
    activation_54 (Activation)      (None, 25, 25, 32)   0           batch_normalization_54[0][0]     
    __________________________________________________________________________________________________
    activation_56 (Activation)      (None, 25, 25, 32)   0           batch_normalization_56[0][0]     
    __________________________________________________________________________________________________
    activation_59 (Activation)      (None, 25, 25, 64)   0           batch_normalization_59[0][0]     
    __________________________________________________________________________________________________
    block35_8_mixed (Concatenate)   (None, 25, 25, 128)  0           activation_54[0][0]              
                                                                     activation_56[0][0]              
                                                                     activation_59[0][0]              
    __________________________________________________________________________________________________
    block35_8_conv (Conv2D)         (None, 25, 25, 320)  41280       block35_8_mixed[0][0]            
    __________________________________________________________________________________________________
    block35_8 (Lambda)              (None, 25, 25, 320)  0           block35_7_ac[0][0]               
                                                                     block35_8_conv[0][0]             
    __________________________________________________________________________________________________
    block35_8_ac (Activation)       (None, 25, 25, 320)  0           block35_8[0][0]                  
    __________________________________________________________________________________________________
    conv2d_63 (Conv2D)              (None, 25, 25, 32)   10240       block35_8_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_63 (BatchNo (None, 25, 25, 32)   96          conv2d_63[0][0]                  
    __________________________________________________________________________________________________
    activation_63 (Activation)      (None, 25, 25, 32)   0           batch_normalization_63[0][0]     
    __________________________________________________________________________________________________
    conv2d_61 (Conv2D)              (None, 25, 25, 32)   10240       block35_8_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_64 (Conv2D)              (None, 25, 25, 48)   13824       activation_63[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_61 (BatchNo (None, 25, 25, 32)   96          conv2d_61[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_64 (BatchNo (None, 25, 25, 48)   144         conv2d_64[0][0]                  
    __________________________________________________________________________________________________
    activation_61 (Activation)      (None, 25, 25, 32)   0           batch_normalization_61[0][0]     
    __________________________________________________________________________________________________
    activation_64 (Activation)      (None, 25, 25, 48)   0           batch_normalization_64[0][0]     
    __________________________________________________________________________________________________
    conv2d_60 (Conv2D)              (None, 25, 25, 32)   10240       block35_8_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_62 (Conv2D)              (None, 25, 25, 32)   9216        activation_61[0][0]              
    __________________________________________________________________________________________________
    conv2d_65 (Conv2D)              (None, 25, 25, 64)   27648       activation_64[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_60 (BatchNo (None, 25, 25, 32)   96          conv2d_60[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_62 (BatchNo (None, 25, 25, 32)   96          conv2d_62[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_65 (BatchNo (None, 25, 25, 64)   192         conv2d_65[0][0]                  
    __________________________________________________________________________________________________
    activation_60 (Activation)      (None, 25, 25, 32)   0           batch_normalization_60[0][0]     
    __________________________________________________________________________________________________
    activation_62 (Activation)      (None, 25, 25, 32)   0           batch_normalization_62[0][0]     
    __________________________________________________________________________________________________
    activation_65 (Activation)      (None, 25, 25, 64)   0           batch_normalization_65[0][0]     
    __________________________________________________________________________________________________
    block35_9_mixed (Concatenate)   (None, 25, 25, 128)  0           activation_60[0][0]              
                                                                     activation_62[0][0]              
                                                                     activation_65[0][0]              
    __________________________________________________________________________________________________
    block35_9_conv (Conv2D)         (None, 25, 25, 320)  41280       block35_9_mixed[0][0]            
    __________________________________________________________________________________________________
    block35_9 (Lambda)              (None, 25, 25, 320)  0           block35_8_ac[0][0]               
                                                                     block35_9_conv[0][0]             
    __________________________________________________________________________________________________
    block35_9_ac (Activation)       (None, 25, 25, 320)  0           block35_9[0][0]                  
    __________________________________________________________________________________________________
    conv2d_69 (Conv2D)              (None, 25, 25, 32)   10240       block35_9_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_69 (BatchNo (None, 25, 25, 32)   96          conv2d_69[0][0]                  
    __________________________________________________________________________________________________
    activation_69 (Activation)      (None, 25, 25, 32)   0           batch_normalization_69[0][0]     
    __________________________________________________________________________________________________
    conv2d_67 (Conv2D)              (None, 25, 25, 32)   10240       block35_9_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_70 (Conv2D)              (None, 25, 25, 48)   13824       activation_69[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_67 (BatchNo (None, 25, 25, 32)   96          conv2d_67[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_70 (BatchNo (None, 25, 25, 48)   144         conv2d_70[0][0]                  
    __________________________________________________________________________________________________
    activation_67 (Activation)      (None, 25, 25, 32)   0           batch_normalization_67[0][0]     
    __________________________________________________________________________________________________
    activation_70 (Activation)      (None, 25, 25, 48)   0           batch_normalization_70[0][0]     
    __________________________________________________________________________________________________
    conv2d_66 (Conv2D)              (None, 25, 25, 32)   10240       block35_9_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_68 (Conv2D)              (None, 25, 25, 32)   9216        activation_67[0][0]              
    __________________________________________________________________________________________________
    conv2d_71 (Conv2D)              (None, 25, 25, 64)   27648       activation_70[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_66 (BatchNo (None, 25, 25, 32)   96          conv2d_66[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_68 (BatchNo (None, 25, 25, 32)   96          conv2d_68[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_71 (BatchNo (None, 25, 25, 64)   192         conv2d_71[0][0]                  
    __________________________________________________________________________________________________
    activation_66 (Activation)      (None, 25, 25, 32)   0           batch_normalization_66[0][0]     
    __________________________________________________________________________________________________
    activation_68 (Activation)      (None, 25, 25, 32)   0           batch_normalization_68[0][0]     
    __________________________________________________________________________________________________
    activation_71 (Activation)      (None, 25, 25, 64)   0           batch_normalization_71[0][0]     
    __________________________________________________________________________________________________
    block35_10_mixed (Concatenate)  (None, 25, 25, 128)  0           activation_66[0][0]              
                                                                     activation_68[0][0]              
                                                                     activation_71[0][0]              
    __________________________________________________________________________________________________
    block35_10_conv (Conv2D)        (None, 25, 25, 320)  41280       block35_10_mixed[0][0]           
    __________________________________________________________________________________________________
    block35_10 (Lambda)             (None, 25, 25, 320)  0           block35_9_ac[0][0]               
                                                                     block35_10_conv[0][0]            
    __________________________________________________________________________________________________
    block35_10_ac (Activation)      (None, 25, 25, 320)  0           block35_10[0][0]                 
    __________________________________________________________________________________________________
    conv2d_73 (Conv2D)              (None, 25, 25, 256)  81920       block35_10_ac[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_73 (BatchNo (None, 25, 25, 256)  768         conv2d_73[0][0]                  
    __________________________________________________________________________________________________
    activation_73 (Activation)      (None, 25, 25, 256)  0           batch_normalization_73[0][0]     
    __________________________________________________________________________________________________
    conv2d_74 (Conv2D)              (None, 25, 25, 256)  589824      activation_73[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_74 (BatchNo (None, 25, 25, 256)  768         conv2d_74[0][0]                  
    __________________________________________________________________________________________________
    activation_74 (Activation)      (None, 25, 25, 256)  0           batch_normalization_74[0][0]     
    __________________________________________________________________________________________________
    conv2d_72 (Conv2D)              (None, 12, 12, 384)  1105920     block35_10_ac[0][0]              
    __________________________________________________________________________________________________
    conv2d_75 (Conv2D)              (None, 12, 12, 384)  884736      activation_74[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_72 (BatchNo (None, 12, 12, 384)  1152        conv2d_72[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_75 (BatchNo (None, 12, 12, 384)  1152        conv2d_75[0][0]                  
    __________________________________________________________________________________________________
    activation_72 (Activation)      (None, 12, 12, 384)  0           batch_normalization_72[0][0]     
    __________________________________________________________________________________________________
    activation_75 (Activation)      (None, 12, 12, 384)  0           batch_normalization_75[0][0]     
    __________________________________________________________________________________________________
    max_pooling2d_2 (MaxPooling2D)  (None, 12, 12, 320)  0           block35_10_ac[0][0]              
    __________________________________________________________________________________________________
    mixed_6a (Concatenate)          (None, 12, 12, 1088) 0           activation_72[0][0]              
                                                                     activation_75[0][0]              
                                                                     max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    conv2d_77 (Conv2D)              (None, 12, 12, 128)  139264      mixed_6a[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_77 (BatchNo (None, 12, 12, 128)  384         conv2d_77[0][0]                  
    __________________________________________________________________________________________________
    activation_77 (Activation)      (None, 12, 12, 128)  0           batch_normalization_77[0][0]     
    __________________________________________________________________________________________________
    conv2d_78 (Conv2D)              (None, 12, 12, 160)  143360      activation_77[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_78 (BatchNo (None, 12, 12, 160)  480         conv2d_78[0][0]                  
    __________________________________________________________________________________________________
    activation_78 (Activation)      (None, 12, 12, 160)  0           batch_normalization_78[0][0]     
    __________________________________________________________________________________________________
    conv2d_76 (Conv2D)              (None, 12, 12, 192)  208896      mixed_6a[0][0]                   
    __________________________________________________________________________________________________
    conv2d_79 (Conv2D)              (None, 12, 12, 192)  215040      activation_78[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_76 (BatchNo (None, 12, 12, 192)  576         conv2d_76[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_79 (BatchNo (None, 12, 12, 192)  576         conv2d_79[0][0]                  
    __________________________________________________________________________________________________
    activation_76 (Activation)      (None, 12, 12, 192)  0           batch_normalization_76[0][0]     
    __________________________________________________________________________________________________
    activation_79 (Activation)      (None, 12, 12, 192)  0           batch_normalization_79[0][0]     
    __________________________________________________________________________________________________
    block17_1_mixed (Concatenate)   (None, 12, 12, 384)  0           activation_76[0][0]              
                                                                     activation_79[0][0]              
    __________________________________________________________________________________________________
    block17_1_conv (Conv2D)         (None, 12, 12, 1088) 418880      block17_1_mixed[0][0]            
    __________________________________________________________________________________________________
    block17_1 (Lambda)              (None, 12, 12, 1088) 0           mixed_6a[0][0]                   
                                                                     block17_1_conv[0][0]             
    __________________________________________________________________________________________________
    block17_1_ac (Activation)       (None, 12, 12, 1088) 0           block17_1[0][0]                  
    __________________________________________________________________________________________________
    conv2d_81 (Conv2D)              (None, 12, 12, 128)  139264      block17_1_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_81 (BatchNo (None, 12, 12, 128)  384         conv2d_81[0][0]                  
    __________________________________________________________________________________________________
    activation_81 (Activation)      (None, 12, 12, 128)  0           batch_normalization_81[0][0]     
    __________________________________________________________________________________________________
    conv2d_82 (Conv2D)              (None, 12, 12, 160)  143360      activation_81[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_82 (BatchNo (None, 12, 12, 160)  480         conv2d_82[0][0]                  
    __________________________________________________________________________________________________
    activation_82 (Activation)      (None, 12, 12, 160)  0           batch_normalization_82[0][0]     
    __________________________________________________________________________________________________
    conv2d_80 (Conv2D)              (None, 12, 12, 192)  208896      block17_1_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_83 (Conv2D)              (None, 12, 12, 192)  215040      activation_82[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_80 (BatchNo (None, 12, 12, 192)  576         conv2d_80[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_83 (BatchNo (None, 12, 12, 192)  576         conv2d_83[0][0]                  
    __________________________________________________________________________________________________
    activation_80 (Activation)      (None, 12, 12, 192)  0           batch_normalization_80[0][0]     
    __________________________________________________________________________________________________
    activation_83 (Activation)      (None, 12, 12, 192)  0           batch_normalization_83[0][0]     
    __________________________________________________________________________________________________
    block17_2_mixed (Concatenate)   (None, 12, 12, 384)  0           activation_80[0][0]              
                                                                     activation_83[0][0]              
    __________________________________________________________________________________________________
    block17_2_conv (Conv2D)         (None, 12, 12, 1088) 418880      block17_2_mixed[0][0]            
    __________________________________________________________________________________________________
    block17_2 (Lambda)              (None, 12, 12, 1088) 0           block17_1_ac[0][0]               
                                                                     block17_2_conv[0][0]             
    __________________________________________________________________________________________________
    block17_2_ac (Activation)       (None, 12, 12, 1088) 0           block17_2[0][0]                  
    __________________________________________________________________________________________________
    conv2d_85 (Conv2D)              (None, 12, 12, 128)  139264      block17_2_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_85 (BatchNo (None, 12, 12, 128)  384         conv2d_85[0][0]                  
    __________________________________________________________________________________________________
    activation_85 (Activation)      (None, 12, 12, 128)  0           batch_normalization_85[0][0]     
    __________________________________________________________________________________________________
    conv2d_86 (Conv2D)              (None, 12, 12, 160)  143360      activation_85[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_86 (BatchNo (None, 12, 12, 160)  480         conv2d_86[0][0]                  
    __________________________________________________________________________________________________
    activation_86 (Activation)      (None, 12, 12, 160)  0           batch_normalization_86[0][0]     
    __________________________________________________________________________________________________
    conv2d_84 (Conv2D)              (None, 12, 12, 192)  208896      block17_2_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_87 (Conv2D)              (None, 12, 12, 192)  215040      activation_86[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_84 (BatchNo (None, 12, 12, 192)  576         conv2d_84[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_87 (BatchNo (None, 12, 12, 192)  576         conv2d_87[0][0]                  
    __________________________________________________________________________________________________
    activation_84 (Activation)      (None, 12, 12, 192)  0           batch_normalization_84[0][0]     
    __________________________________________________________________________________________________
    activation_87 (Activation)      (None, 12, 12, 192)  0           batch_normalization_87[0][0]     
    __________________________________________________________________________________________________
    block17_3_mixed (Concatenate)   (None, 12, 12, 384)  0           activation_84[0][0]              
                                                                     activation_87[0][0]              
    __________________________________________________________________________________________________
    block17_3_conv (Conv2D)         (None, 12, 12, 1088) 418880      block17_3_mixed[0][0]            
    __________________________________________________________________________________________________
    block17_3 (Lambda)              (None, 12, 12, 1088) 0           block17_2_ac[0][0]               
                                                                     block17_3_conv[0][0]             
    __________________________________________________________________________________________________
    block17_3_ac (Activation)       (None, 12, 12, 1088) 0           block17_3[0][0]                  
    __________________________________________________________________________________________________
    conv2d_89 (Conv2D)              (None, 12, 12, 128)  139264      block17_3_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_89 (BatchNo (None, 12, 12, 128)  384         conv2d_89[0][0]                  
    __________________________________________________________________________________________________
    activation_89 (Activation)      (None, 12, 12, 128)  0           batch_normalization_89[0][0]     
    __________________________________________________________________________________________________
    conv2d_90 (Conv2D)              (None, 12, 12, 160)  143360      activation_89[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_90 (BatchNo (None, 12, 12, 160)  480         conv2d_90[0][0]                  
    __________________________________________________________________________________________________
    activation_90 (Activation)      (None, 12, 12, 160)  0           batch_normalization_90[0][0]     
    __________________________________________________________________________________________________
    conv2d_88 (Conv2D)              (None, 12, 12, 192)  208896      block17_3_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_91 (Conv2D)              (None, 12, 12, 192)  215040      activation_90[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_88 (BatchNo (None, 12, 12, 192)  576         conv2d_88[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_91 (BatchNo (None, 12, 12, 192)  576         conv2d_91[0][0]                  
    __________________________________________________________________________________________________
    activation_88 (Activation)      (None, 12, 12, 192)  0           batch_normalization_88[0][0]     
    __________________________________________________________________________________________________
    activation_91 (Activation)      (None, 12, 12, 192)  0           batch_normalization_91[0][0]     
    __________________________________________________________________________________________________
    block17_4_mixed (Concatenate)   (None, 12, 12, 384)  0           activation_88[0][0]              
                                                                     activation_91[0][0]              
    __________________________________________________________________________________________________
    block17_4_conv (Conv2D)         (None, 12, 12, 1088) 418880      block17_4_mixed[0][0]            
    __________________________________________________________________________________________________
    block17_4 (Lambda)              (None, 12, 12, 1088) 0           block17_3_ac[0][0]               
                                                                     block17_4_conv[0][0]             
    __________________________________________________________________________________________________
    block17_4_ac (Activation)       (None, 12, 12, 1088) 0           block17_4[0][0]                  
    __________________________________________________________________________________________________
    conv2d_93 (Conv2D)              (None, 12, 12, 128)  139264      block17_4_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_93 (BatchNo (None, 12, 12, 128)  384         conv2d_93[0][0]                  
    __________________________________________________________________________________________________
    activation_93 (Activation)      (None, 12, 12, 128)  0           batch_normalization_93[0][0]     
    __________________________________________________________________________________________________
    conv2d_94 (Conv2D)              (None, 12, 12, 160)  143360      activation_93[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_94 (BatchNo (None, 12, 12, 160)  480         conv2d_94[0][0]                  
    __________________________________________________________________________________________________
    activation_94 (Activation)      (None, 12, 12, 160)  0           batch_normalization_94[0][0]     
    __________________________________________________________________________________________________
    conv2d_92 (Conv2D)              (None, 12, 12, 192)  208896      block17_4_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_95 (Conv2D)              (None, 12, 12, 192)  215040      activation_94[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_92 (BatchNo (None, 12, 12, 192)  576         conv2d_92[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_95 (BatchNo (None, 12, 12, 192)  576         conv2d_95[0][0]                  
    __________________________________________________________________________________________________
    activation_92 (Activation)      (None, 12, 12, 192)  0           batch_normalization_92[0][0]     
    __________________________________________________________________________________________________
    activation_95 (Activation)      (None, 12, 12, 192)  0           batch_normalization_95[0][0]     
    __________________________________________________________________________________________________
    block17_5_mixed (Concatenate)   (None, 12, 12, 384)  0           activation_92[0][0]              
                                                                     activation_95[0][0]              
    __________________________________________________________________________________________________
    block17_5_conv (Conv2D)         (None, 12, 12, 1088) 418880      block17_5_mixed[0][0]            
    __________________________________________________________________________________________________
    block17_5 (Lambda)              (None, 12, 12, 1088) 0           block17_4_ac[0][0]               
                                                                     block17_5_conv[0][0]             
    __________________________________________________________________________________________________
    block17_5_ac (Activation)       (None, 12, 12, 1088) 0           block17_5[0][0]                  
    __________________________________________________________________________________________________
    conv2d_97 (Conv2D)              (None, 12, 12, 128)  139264      block17_5_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_97 (BatchNo (None, 12, 12, 128)  384         conv2d_97[0][0]                  
    __________________________________________________________________________________________________
    activation_97 (Activation)      (None, 12, 12, 128)  0           batch_normalization_97[0][0]     
    __________________________________________________________________________________________________
    conv2d_98 (Conv2D)              (None, 12, 12, 160)  143360      activation_97[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_98 (BatchNo (None, 12, 12, 160)  480         conv2d_98[0][0]                  
    __________________________________________________________________________________________________
    activation_98 (Activation)      (None, 12, 12, 160)  0           batch_normalization_98[0][0]     
    __________________________________________________________________________________________________
    conv2d_96 (Conv2D)              (None, 12, 12, 192)  208896      block17_5_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_99 (Conv2D)              (None, 12, 12, 192)  215040      activation_98[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_96 (BatchNo (None, 12, 12, 192)  576         conv2d_96[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_99 (BatchNo (None, 12, 12, 192)  576         conv2d_99[0][0]                  
    __________________________________________________________________________________________________
    activation_96 (Activation)      (None, 12, 12, 192)  0           batch_normalization_96[0][0]     
    __________________________________________________________________________________________________
    activation_99 (Activation)      (None, 12, 12, 192)  0           batch_normalization_99[0][0]     
    __________________________________________________________________________________________________
    block17_6_mixed (Concatenate)   (None, 12, 12, 384)  0           activation_96[0][0]              
                                                                     activation_99[0][0]              
    __________________________________________________________________________________________________
    block17_6_conv (Conv2D)         (None, 12, 12, 1088) 418880      block17_6_mixed[0][0]            
    __________________________________________________________________________________________________
    block17_6 (Lambda)              (None, 12, 12, 1088) 0           block17_5_ac[0][0]               
                                                                     block17_6_conv[0][0]             
    __________________________________________________________________________________________________
    block17_6_ac (Activation)       (None, 12, 12, 1088) 0           block17_6[0][0]                  
    __________________________________________________________________________________________________
    conv2d_101 (Conv2D)             (None, 12, 12, 128)  139264      block17_6_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_101 (BatchN (None, 12, 12, 128)  384         conv2d_101[0][0]                 
    __________________________________________________________________________________________________
    activation_101 (Activation)     (None, 12, 12, 128)  0           batch_normalization_101[0][0]    
    __________________________________________________________________________________________________
    conv2d_102 (Conv2D)             (None, 12, 12, 160)  143360      activation_101[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_102 (BatchN (None, 12, 12, 160)  480         conv2d_102[0][0]                 
    __________________________________________________________________________________________________
    activation_102 (Activation)     (None, 12, 12, 160)  0           batch_normalization_102[0][0]    
    __________________________________________________________________________________________________
    conv2d_100 (Conv2D)             (None, 12, 12, 192)  208896      block17_6_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_103 (Conv2D)             (None, 12, 12, 192)  215040      activation_102[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_100 (BatchN (None, 12, 12, 192)  576         conv2d_100[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_103 (BatchN (None, 12, 12, 192)  576         conv2d_103[0][0]                 
    __________________________________________________________________________________________________
    activation_100 (Activation)     (None, 12, 12, 192)  0           batch_normalization_100[0][0]    
    __________________________________________________________________________________________________
    activation_103 (Activation)     (None, 12, 12, 192)  0           batch_normalization_103[0][0]    
    __________________________________________________________________________________________________
    block17_7_mixed (Concatenate)   (None, 12, 12, 384)  0           activation_100[0][0]             
                                                                     activation_103[0][0]             
    __________________________________________________________________________________________________
    block17_7_conv (Conv2D)         (None, 12, 12, 1088) 418880      block17_7_mixed[0][0]            
    __________________________________________________________________________________________________
    block17_7 (Lambda)              (None, 12, 12, 1088) 0           block17_6_ac[0][0]               
                                                                     block17_7_conv[0][0]             
    __________________________________________________________________________________________________
    block17_7_ac (Activation)       (None, 12, 12, 1088) 0           block17_7[0][0]                  
    __________________________________________________________________________________________________
    conv2d_105 (Conv2D)             (None, 12, 12, 128)  139264      block17_7_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_105 (BatchN (None, 12, 12, 128)  384         conv2d_105[0][0]                 
    __________________________________________________________________________________________________
    activation_105 (Activation)     (None, 12, 12, 128)  0           batch_normalization_105[0][0]    
    __________________________________________________________________________________________________
    conv2d_106 (Conv2D)             (None, 12, 12, 160)  143360      activation_105[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_106 (BatchN (None, 12, 12, 160)  480         conv2d_106[0][0]                 
    __________________________________________________________________________________________________
    activation_106 (Activation)     (None, 12, 12, 160)  0           batch_normalization_106[0][0]    
    __________________________________________________________________________________________________
    conv2d_104 (Conv2D)             (None, 12, 12, 192)  208896      block17_7_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_107 (Conv2D)             (None, 12, 12, 192)  215040      activation_106[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_104 (BatchN (None, 12, 12, 192)  576         conv2d_104[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_107 (BatchN (None, 12, 12, 192)  576         conv2d_107[0][0]                 
    __________________________________________________________________________________________________
    activation_104 (Activation)     (None, 12, 12, 192)  0           batch_normalization_104[0][0]    
    __________________________________________________________________________________________________
    activation_107 (Activation)     (None, 12, 12, 192)  0           batch_normalization_107[0][0]    
    __________________________________________________________________________________________________
    block17_8_mixed (Concatenate)   (None, 12, 12, 384)  0           activation_104[0][0]             
                                                                     activation_107[0][0]             
    __________________________________________________________________________________________________
    block17_8_conv (Conv2D)         (None, 12, 12, 1088) 418880      block17_8_mixed[0][0]            
    __________________________________________________________________________________________________
    block17_8 (Lambda)              (None, 12, 12, 1088) 0           block17_7_ac[0][0]               
                                                                     block17_8_conv[0][0]             
    __________________________________________________________________________________________________
    block17_8_ac (Activation)       (None, 12, 12, 1088) 0           block17_8[0][0]                  
    __________________________________________________________________________________________________
    conv2d_109 (Conv2D)             (None, 12, 12, 128)  139264      block17_8_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_109 (BatchN (None, 12, 12, 128)  384         conv2d_109[0][0]                 
    __________________________________________________________________________________________________
    activation_109 (Activation)     (None, 12, 12, 128)  0           batch_normalization_109[0][0]    
    __________________________________________________________________________________________________
    conv2d_110 (Conv2D)             (None, 12, 12, 160)  143360      activation_109[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_110 (BatchN (None, 12, 12, 160)  480         conv2d_110[0][0]                 
    __________________________________________________________________________________________________
    activation_110 (Activation)     (None, 12, 12, 160)  0           batch_normalization_110[0][0]    
    __________________________________________________________________________________________________
    conv2d_108 (Conv2D)             (None, 12, 12, 192)  208896      block17_8_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_111 (Conv2D)             (None, 12, 12, 192)  215040      activation_110[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_108 (BatchN (None, 12, 12, 192)  576         conv2d_108[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_111 (BatchN (None, 12, 12, 192)  576         conv2d_111[0][0]                 
    __________________________________________________________________________________________________
    activation_108 (Activation)     (None, 12, 12, 192)  0           batch_normalization_108[0][0]    
    __________________________________________________________________________________________________
    activation_111 (Activation)     (None, 12, 12, 192)  0           batch_normalization_111[0][0]    
    __________________________________________________________________________________________________
    block17_9_mixed (Concatenate)   (None, 12, 12, 384)  0           activation_108[0][0]             
                                                                     activation_111[0][0]             
    __________________________________________________________________________________________________
    block17_9_conv (Conv2D)         (None, 12, 12, 1088) 418880      block17_9_mixed[0][0]            
    __________________________________________________________________________________________________
    block17_9 (Lambda)              (None, 12, 12, 1088) 0           block17_8_ac[0][0]               
                                                                     block17_9_conv[0][0]             
    __________________________________________________________________________________________________
    block17_9_ac (Activation)       (None, 12, 12, 1088) 0           block17_9[0][0]                  
    __________________________________________________________________________________________________
    conv2d_113 (Conv2D)             (None, 12, 12, 128)  139264      block17_9_ac[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_113 (BatchN (None, 12, 12, 128)  384         conv2d_113[0][0]                 
    __________________________________________________________________________________________________
    activation_113 (Activation)     (None, 12, 12, 128)  0           batch_normalization_113[0][0]    
    __________________________________________________________________________________________________
    conv2d_114 (Conv2D)             (None, 12, 12, 160)  143360      activation_113[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_114 (BatchN (None, 12, 12, 160)  480         conv2d_114[0][0]                 
    __________________________________________________________________________________________________
    activation_114 (Activation)     (None, 12, 12, 160)  0           batch_normalization_114[0][0]    
    __________________________________________________________________________________________________
    conv2d_112 (Conv2D)             (None, 12, 12, 192)  208896      block17_9_ac[0][0]               
    __________________________________________________________________________________________________
    conv2d_115 (Conv2D)             (None, 12, 12, 192)  215040      activation_114[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_112 (BatchN (None, 12, 12, 192)  576         conv2d_112[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_115 (BatchN (None, 12, 12, 192)  576         conv2d_115[0][0]                 
    __________________________________________________________________________________________________
    activation_112 (Activation)     (None, 12, 12, 192)  0           batch_normalization_112[0][0]    
    __________________________________________________________________________________________________
    activation_115 (Activation)     (None, 12, 12, 192)  0           batch_normalization_115[0][0]    
    __________________________________________________________________________________________________
    block17_10_mixed (Concatenate)  (None, 12, 12, 384)  0           activation_112[0][0]             
                                                                     activation_115[0][0]             
    __________________________________________________________________________________________________
    block17_10_conv (Conv2D)        (None, 12, 12, 1088) 418880      block17_10_mixed[0][0]           
    __________________________________________________________________________________________________
    block17_10 (Lambda)             (None, 12, 12, 1088) 0           block17_9_ac[0][0]               
                                                                     block17_10_conv[0][0]            
    __________________________________________________________________________________________________
    block17_10_ac (Activation)      (None, 12, 12, 1088) 0           block17_10[0][0]                 
    __________________________________________________________________________________________________
    conv2d_117 (Conv2D)             (None, 12, 12, 128)  139264      block17_10_ac[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_117 (BatchN (None, 12, 12, 128)  384         conv2d_117[0][0]                 
    __________________________________________________________________________________________________
    activation_117 (Activation)     (None, 12, 12, 128)  0           batch_normalization_117[0][0]    
    __________________________________________________________________________________________________
    conv2d_118 (Conv2D)             (None, 12, 12, 160)  143360      activation_117[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_118 (BatchN (None, 12, 12, 160)  480         conv2d_118[0][0]                 
    __________________________________________________________________________________________________
    activation_118 (Activation)     (None, 12, 12, 160)  0           batch_normalization_118[0][0]    
    __________________________________________________________________________________________________
    conv2d_116 (Conv2D)             (None, 12, 12, 192)  208896      block17_10_ac[0][0]              
    __________________________________________________________________________________________________
    conv2d_119 (Conv2D)             (None, 12, 12, 192)  215040      activation_118[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_116 (BatchN (None, 12, 12, 192)  576         conv2d_116[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_119 (BatchN (None, 12, 12, 192)  576         conv2d_119[0][0]                 
    __________________________________________________________________________________________________
    activation_116 (Activation)     (None, 12, 12, 192)  0           batch_normalization_116[0][0]    
    __________________________________________________________________________________________________
    activation_119 (Activation)     (None, 12, 12, 192)  0           batch_normalization_119[0][0]    
    __________________________________________________________________________________________________
    block17_11_mixed (Concatenate)  (None, 12, 12, 384)  0           activation_116[0][0]             
                                                                     activation_119[0][0]             
    __________________________________________________________________________________________________
    block17_11_conv (Conv2D)        (None, 12, 12, 1088) 418880      block17_11_mixed[0][0]           
    __________________________________________________________________________________________________
    block17_11 (Lambda)             (None, 12, 12, 1088) 0           block17_10_ac[0][0]              
                                                                     block17_11_conv[0][0]            
    __________________________________________________________________________________________________
    block17_11_ac (Activation)      (None, 12, 12, 1088) 0           block17_11[0][0]                 
    __________________________________________________________________________________________________
    conv2d_121 (Conv2D)             (None, 12, 12, 128)  139264      block17_11_ac[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_121 (BatchN (None, 12, 12, 128)  384         conv2d_121[0][0]                 
    __________________________________________________________________________________________________
    activation_121 (Activation)     (None, 12, 12, 128)  0           batch_normalization_121[0][0]    
    __________________________________________________________________________________________________
    conv2d_122 (Conv2D)             (None, 12, 12, 160)  143360      activation_121[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_122 (BatchN (None, 12, 12, 160)  480         conv2d_122[0][0]                 
    __________________________________________________________________________________________________
    activation_122 (Activation)     (None, 12, 12, 160)  0           batch_normalization_122[0][0]    
    __________________________________________________________________________________________________
    conv2d_120 (Conv2D)             (None, 12, 12, 192)  208896      block17_11_ac[0][0]              
    __________________________________________________________________________________________________
    conv2d_123 (Conv2D)             (None, 12, 12, 192)  215040      activation_122[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_120 (BatchN (None, 12, 12, 192)  576         conv2d_120[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_123 (BatchN (None, 12, 12, 192)  576         conv2d_123[0][0]                 
    __________________________________________________________________________________________________
    activation_120 (Activation)     (None, 12, 12, 192)  0           batch_normalization_120[0][0]    
    __________________________________________________________________________________________________
    activation_123 (Activation)     (None, 12, 12, 192)  0           batch_normalization_123[0][0]    
    __________________________________________________________________________________________________
    block17_12_mixed (Concatenate)  (None, 12, 12, 384)  0           activation_120[0][0]             
                                                                     activation_123[0][0]             
    __________________________________________________________________________________________________
    block17_12_conv (Conv2D)        (None, 12, 12, 1088) 418880      block17_12_mixed[0][0]           
    __________________________________________________________________________________________________
    block17_12 (Lambda)             (None, 12, 12, 1088) 0           block17_11_ac[0][0]              
                                                                     block17_12_conv[0][0]            
    __________________________________________________________________________________________________
    block17_12_ac (Activation)      (None, 12, 12, 1088) 0           block17_12[0][0]                 
    __________________________________________________________________________________________________
    conv2d_125 (Conv2D)             (None, 12, 12, 128)  139264      block17_12_ac[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_125 (BatchN (None, 12, 12, 128)  384         conv2d_125[0][0]                 
    __________________________________________________________________________________________________
    activation_125 (Activation)     (None, 12, 12, 128)  0           batch_normalization_125[0][0]    
    __________________________________________________________________________________________________
    conv2d_126 (Conv2D)             (None, 12, 12, 160)  143360      activation_125[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_126 (BatchN (None, 12, 12, 160)  480         conv2d_126[0][0]                 
    __________________________________________________________________________________________________
    activation_126 (Activation)     (None, 12, 12, 160)  0           batch_normalization_126[0][0]    
    __________________________________________________________________________________________________
    conv2d_124 (Conv2D)             (None, 12, 12, 192)  208896      block17_12_ac[0][0]              
    __________________________________________________________________________________________________
    conv2d_127 (Conv2D)             (None, 12, 12, 192)  215040      activation_126[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_124 (BatchN (None, 12, 12, 192)  576         conv2d_124[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_127 (BatchN (None, 12, 12, 192)  576         conv2d_127[0][0]                 
    __________________________________________________________________________________________________
    activation_124 (Activation)     (None, 12, 12, 192)  0           batch_normalization_124[0][0]    
    __________________________________________________________________________________________________
    activation_127 (Activation)     (None, 12, 12, 192)  0           batch_normalization_127[0][0]    
    __________________________________________________________________________________________________
    block17_13_mixed (Concatenate)  (None, 12, 12, 384)  0           activation_124[0][0]             
                                                                     activation_127[0][0]             
    __________________________________________________________________________________________________
    block17_13_conv (Conv2D)        (None, 12, 12, 1088) 418880      block17_13_mixed[0][0]           
    __________________________________________________________________________________________________
    block17_13 (Lambda)             (None, 12, 12, 1088) 0           block17_12_ac[0][0]              
                                                                     block17_13_conv[0][0]            
    __________________________________________________________________________________________________
    block17_13_ac (Activation)      (None, 12, 12, 1088) 0           block17_13[0][0]                 
    __________________________________________________________________________________________________
    conv2d_129 (Conv2D)             (None, 12, 12, 128)  139264      block17_13_ac[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_129 (BatchN (None, 12, 12, 128)  384         conv2d_129[0][0]                 
    __________________________________________________________________________________________________
    activation_129 (Activation)     (None, 12, 12, 128)  0           batch_normalization_129[0][0]    
    __________________________________________________________________________________________________
    conv2d_130 (Conv2D)             (None, 12, 12, 160)  143360      activation_129[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_130 (BatchN (None, 12, 12, 160)  480         conv2d_130[0][0]                 
    __________________________________________________________________________________________________
    activation_130 (Activation)     (None, 12, 12, 160)  0           batch_normalization_130[0][0]    
    __________________________________________________________________________________________________
    conv2d_128 (Conv2D)             (None, 12, 12, 192)  208896      block17_13_ac[0][0]              
    __________________________________________________________________________________________________
    conv2d_131 (Conv2D)             (None, 12, 12, 192)  215040      activation_130[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_128 (BatchN (None, 12, 12, 192)  576         conv2d_128[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_131 (BatchN (None, 12, 12, 192)  576         conv2d_131[0][0]                 
    __________________________________________________________________________________________________
    activation_128 (Activation)     (None, 12, 12, 192)  0           batch_normalization_128[0][0]    
    __________________________________________________________________________________________________
    activation_131 (Activation)     (None, 12, 12, 192)  0           batch_normalization_131[0][0]    
    __________________________________________________________________________________________________
    block17_14_mixed (Concatenate)  (None, 12, 12, 384)  0           activation_128[0][0]             
                                                                     activation_131[0][0]             
    __________________________________________________________________________________________________
    block17_14_conv (Conv2D)        (None, 12, 12, 1088) 418880      block17_14_mixed[0][0]           
    __________________________________________________________________________________________________
    block17_14 (Lambda)             (None, 12, 12, 1088) 0           block17_13_ac[0][0]              
                                                                     block17_14_conv[0][0]            
    __________________________________________________________________________________________________
    block17_14_ac (Activation)      (None, 12, 12, 1088) 0           block17_14[0][0]                 
    __________________________________________________________________________________________________
    conv2d_133 (Conv2D)             (None, 12, 12, 128)  139264      block17_14_ac[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_133 (BatchN (None, 12, 12, 128)  384         conv2d_133[0][0]                 
    __________________________________________________________________________________________________
    activation_133 (Activation)     (None, 12, 12, 128)  0           batch_normalization_133[0][0]    
    __________________________________________________________________________________________________
    conv2d_134 (Conv2D)             (None, 12, 12, 160)  143360      activation_133[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_134 (BatchN (None, 12, 12, 160)  480         conv2d_134[0][0]                 
    __________________________________________________________________________________________________
    activation_134 (Activation)     (None, 12, 12, 160)  0           batch_normalization_134[0][0]    
    __________________________________________________________________________________________________
    conv2d_132 (Conv2D)             (None, 12, 12, 192)  208896      block17_14_ac[0][0]              
    __________________________________________________________________________________________________
    conv2d_135 (Conv2D)             (None, 12, 12, 192)  215040      activation_134[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_132 (BatchN (None, 12, 12, 192)  576         conv2d_132[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_135 (BatchN (None, 12, 12, 192)  576         conv2d_135[0][0]                 
    __________________________________________________________________________________________________
    activation_132 (Activation)     (None, 12, 12, 192)  0           batch_normalization_132[0][0]    
    __________________________________________________________________________________________________
    activation_135 (Activation)     (None, 12, 12, 192)  0           batch_normalization_135[0][0]    
    __________________________________________________________________________________________________
    block17_15_mixed (Concatenate)  (None, 12, 12, 384)  0           activation_132[0][0]             
                                                                     activation_135[0][0]             
    __________________________________________________________________________________________________
    block17_15_conv (Conv2D)        (None, 12, 12, 1088) 418880      block17_15_mixed[0][0]           
    __________________________________________________________________________________________________
    block17_15 (Lambda)             (None, 12, 12, 1088) 0           block17_14_ac[0][0]              
                                                                     block17_15_conv[0][0]            
    __________________________________________________________________________________________________
    block17_15_ac (Activation)      (None, 12, 12, 1088) 0           block17_15[0][0]                 
    __________________________________________________________________________________________________
    conv2d_137 (Conv2D)             (None, 12, 12, 128)  139264      block17_15_ac[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_137 (BatchN (None, 12, 12, 128)  384         conv2d_137[0][0]                 
    __________________________________________________________________________________________________
    activation_137 (Activation)     (None, 12, 12, 128)  0           batch_normalization_137[0][0]    
    __________________________________________________________________________________________________
    conv2d_138 (Conv2D)             (None, 12, 12, 160)  143360      activation_137[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_138 (BatchN (None, 12, 12, 160)  480         conv2d_138[0][0]                 
    __________________________________________________________________________________________________
    activation_138 (Activation)     (None, 12, 12, 160)  0           batch_normalization_138[0][0]    
    __________________________________________________________________________________________________
    conv2d_136 (Conv2D)             (None, 12, 12, 192)  208896      block17_15_ac[0][0]              
    __________________________________________________________________________________________________
    conv2d_139 (Conv2D)             (None, 12, 12, 192)  215040      activation_138[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_136 (BatchN (None, 12, 12, 192)  576         conv2d_136[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_139 (BatchN (None, 12, 12, 192)  576         conv2d_139[0][0]                 
    __________________________________________________________________________________________________
    activation_136 (Activation)     (None, 12, 12, 192)  0           batch_normalization_136[0][0]    
    __________________________________________________________________________________________________
    activation_139 (Activation)     (None, 12, 12, 192)  0           batch_normalization_139[0][0]    
    __________________________________________________________________________________________________
    block17_16_mixed (Concatenate)  (None, 12, 12, 384)  0           activation_136[0][0]             
                                                                     activation_139[0][0]             
    __________________________________________________________________________________________________
    block17_16_conv (Conv2D)        (None, 12, 12, 1088) 418880      block17_16_mixed[0][0]           
    __________________________________________________________________________________________________
    block17_16 (Lambda)             (None, 12, 12, 1088) 0           block17_15_ac[0][0]              
                                                                     block17_16_conv[0][0]            
    __________________________________________________________________________________________________
    block17_16_ac (Activation)      (None, 12, 12, 1088) 0           block17_16[0][0]                 
    __________________________________________________________________________________________________
    conv2d_141 (Conv2D)             (None, 12, 12, 128)  139264      block17_16_ac[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_141 (BatchN (None, 12, 12, 128)  384         conv2d_141[0][0]                 
    __________________________________________________________________________________________________
    activation_141 (Activation)     (None, 12, 12, 128)  0           batch_normalization_141[0][0]    
    __________________________________________________________________________________________________
    conv2d_142 (Conv2D)             (None, 12, 12, 160)  143360      activation_141[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_142 (BatchN (None, 12, 12, 160)  480         conv2d_142[0][0]                 
    __________________________________________________________________________________________________
    activation_142 (Activation)     (None, 12, 12, 160)  0           batch_normalization_142[0][0]    
    __________________________________________________________________________________________________
    conv2d_140 (Conv2D)             (None, 12, 12, 192)  208896      block17_16_ac[0][0]              
    __________________________________________________________________________________________________
    conv2d_143 (Conv2D)             (None, 12, 12, 192)  215040      activation_142[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_140 (BatchN (None, 12, 12, 192)  576         conv2d_140[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_143 (BatchN (None, 12, 12, 192)  576         conv2d_143[0][0]                 
    __________________________________________________________________________________________________
    activation_140 (Activation)     (None, 12, 12, 192)  0           batch_normalization_140[0][0]    
    __________________________________________________________________________________________________
    activation_143 (Activation)     (None, 12, 12, 192)  0           batch_normalization_143[0][0]    
    __________________________________________________________________________________________________
    block17_17_mixed (Concatenate)  (None, 12, 12, 384)  0           activation_140[0][0]             
                                                                     activation_143[0][0]             
    __________________________________________________________________________________________________
    block17_17_conv (Conv2D)        (None, 12, 12, 1088) 418880      block17_17_mixed[0][0]           
    __________________________________________________________________________________________________
    block17_17 (Lambda)             (None, 12, 12, 1088) 0           block17_16_ac[0][0]              
                                                                     block17_17_conv[0][0]            
    __________________________________________________________________________________________________
    block17_17_ac (Activation)      (None, 12, 12, 1088) 0           block17_17[0][0]                 
    __________________________________________________________________________________________________
    conv2d_145 (Conv2D)             (None, 12, 12, 128)  139264      block17_17_ac[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_145 (BatchN (None, 12, 12, 128)  384         conv2d_145[0][0]                 
    __________________________________________________________________________________________________
    activation_145 (Activation)     (None, 12, 12, 128)  0           batch_normalization_145[0][0]    
    __________________________________________________________________________________________________
    conv2d_146 (Conv2D)             (None, 12, 12, 160)  143360      activation_145[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_146 (BatchN (None, 12, 12, 160)  480         conv2d_146[0][0]                 
    __________________________________________________________________________________________________
    activation_146 (Activation)     (None, 12, 12, 160)  0           batch_normalization_146[0][0]    
    __________________________________________________________________________________________________
    conv2d_144 (Conv2D)             (None, 12, 12, 192)  208896      block17_17_ac[0][0]              
    __________________________________________________________________________________________________
    conv2d_147 (Conv2D)             (None, 12, 12, 192)  215040      activation_146[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_144 (BatchN (None, 12, 12, 192)  576         conv2d_144[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_147 (BatchN (None, 12, 12, 192)  576         conv2d_147[0][0]                 
    __________________________________________________________________________________________________
    activation_144 (Activation)     (None, 12, 12, 192)  0           batch_normalization_144[0][0]    
    __________________________________________________________________________________________________
    activation_147 (Activation)     (None, 12, 12, 192)  0           batch_normalization_147[0][0]    
    __________________________________________________________________________________________________
    block17_18_mixed (Concatenate)  (None, 12, 12, 384)  0           activation_144[0][0]             
                                                                     activation_147[0][0]             
    __________________________________________________________________________________________________
    block17_18_conv (Conv2D)        (None, 12, 12, 1088) 418880      block17_18_mixed[0][0]           
    __________________________________________________________________________________________________
    block17_18 (Lambda)             (None, 12, 12, 1088) 0           block17_17_ac[0][0]              
                                                                     block17_18_conv[0][0]            
    __________________________________________________________________________________________________
    block17_18_ac (Activation)      (None, 12, 12, 1088) 0           block17_18[0][0]                 
    __________________________________________________________________________________________________
    conv2d_149 (Conv2D)             (None, 12, 12, 128)  139264      block17_18_ac[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_149 (BatchN (None, 12, 12, 128)  384         conv2d_149[0][0]                 
    __________________________________________________________________________________________________
    activation_149 (Activation)     (None, 12, 12, 128)  0           batch_normalization_149[0][0]    
    __________________________________________________________________________________________________
    conv2d_150 (Conv2D)             (None, 12, 12, 160)  143360      activation_149[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_150 (BatchN (None, 12, 12, 160)  480         conv2d_150[0][0]                 
    __________________________________________________________________________________________________
    activation_150 (Activation)     (None, 12, 12, 160)  0           batch_normalization_150[0][0]    
    __________________________________________________________________________________________________
    conv2d_148 (Conv2D)             (None, 12, 12, 192)  208896      block17_18_ac[0][0]              
    __________________________________________________________________________________________________
    conv2d_151 (Conv2D)             (None, 12, 12, 192)  215040      activation_150[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_148 (BatchN (None, 12, 12, 192)  576         conv2d_148[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_151 (BatchN (None, 12, 12, 192)  576         conv2d_151[0][0]                 
    __________________________________________________________________________________________________
    activation_148 (Activation)     (None, 12, 12, 192)  0           batch_normalization_148[0][0]    
    __________________________________________________________________________________________________
    activation_151 (Activation)     (None, 12, 12, 192)  0           batch_normalization_151[0][0]    
    __________________________________________________________________________________________________
    block17_19_mixed (Concatenate)  (None, 12, 12, 384)  0           activation_148[0][0]             
                                                                     activation_151[0][0]             
    __________________________________________________________________________________________________
    block17_19_conv (Conv2D)        (None, 12, 12, 1088) 418880      block17_19_mixed[0][0]           
    __________________________________________________________________________________________________
    block17_19 (Lambda)             (None, 12, 12, 1088) 0           block17_18_ac[0][0]              
                                                                     block17_19_conv[0][0]            
    __________________________________________________________________________________________________
    block17_19_ac (Activation)      (None, 12, 12, 1088) 0           block17_19[0][0]                 
    __________________________________________________________________________________________________
    conv2d_153 (Conv2D)             (None, 12, 12, 128)  139264      block17_19_ac[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_153 (BatchN (None, 12, 12, 128)  384         conv2d_153[0][0]                 
    __________________________________________________________________________________________________
    activation_153 (Activation)     (None, 12, 12, 128)  0           batch_normalization_153[0][0]    
    __________________________________________________________________________________________________
    conv2d_154 (Conv2D)             (None, 12, 12, 160)  143360      activation_153[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_154 (BatchN (None, 12, 12, 160)  480         conv2d_154[0][0]                 
    __________________________________________________________________________________________________
    activation_154 (Activation)     (None, 12, 12, 160)  0           batch_normalization_154[0][0]    
    __________________________________________________________________________________________________
    conv2d_152 (Conv2D)             (None, 12, 12, 192)  208896      block17_19_ac[0][0]              
    __________________________________________________________________________________________________
    conv2d_155 (Conv2D)             (None, 12, 12, 192)  215040      activation_154[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_152 (BatchN (None, 12, 12, 192)  576         conv2d_152[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_155 (BatchN (None, 12, 12, 192)  576         conv2d_155[0][0]                 
    __________________________________________________________________________________________________
    activation_152 (Activation)     (None, 12, 12, 192)  0           batch_normalization_152[0][0]    
    __________________________________________________________________________________________________
    activation_155 (Activation)     (None, 12, 12, 192)  0           batch_normalization_155[0][0]    
    __________________________________________________________________________________________________
    block17_20_mixed (Concatenate)  (None, 12, 12, 384)  0           activation_152[0][0]             
                                                                     activation_155[0][0]             
    __________________________________________________________________________________________________
    block17_20_conv (Conv2D)        (None, 12, 12, 1088) 418880      block17_20_mixed[0][0]           
    __________________________________________________________________________________________________
    block17_20 (Lambda)             (None, 12, 12, 1088) 0           block17_19_ac[0][0]              
                                                                     block17_20_conv[0][0]            
    __________________________________________________________________________________________________
    block17_20_ac (Activation)      (None, 12, 12, 1088) 0           block17_20[0][0]                 
    __________________________________________________________________________________________________
    conv2d_160 (Conv2D)             (None, 12, 12, 256)  278528      block17_20_ac[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_160 (BatchN (None, 12, 12, 256)  768         conv2d_160[0][0]                 
    __________________________________________________________________________________________________
    activation_160 (Activation)     (None, 12, 12, 256)  0           batch_normalization_160[0][0]    
    __________________________________________________________________________________________________
    conv2d_156 (Conv2D)             (None, 12, 12, 256)  278528      block17_20_ac[0][0]              
    __________________________________________________________________________________________________
    conv2d_158 (Conv2D)             (None, 12, 12, 256)  278528      block17_20_ac[0][0]              
    __________________________________________________________________________________________________
    conv2d_161 (Conv2D)             (None, 12, 12, 288)  663552      activation_160[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_156 (BatchN (None, 12, 12, 256)  768         conv2d_156[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_158 (BatchN (None, 12, 12, 256)  768         conv2d_158[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_161 (BatchN (None, 12, 12, 288)  864         conv2d_161[0][0]                 
    __________________________________________________________________________________________________
    activation_156 (Activation)     (None, 12, 12, 256)  0           batch_normalization_156[0][0]    
    __________________________________________________________________________________________________
    activation_158 (Activation)     (None, 12, 12, 256)  0           batch_normalization_158[0][0]    
    __________________________________________________________________________________________________
    activation_161 (Activation)     (None, 12, 12, 288)  0           batch_normalization_161[0][0]    
    __________________________________________________________________________________________________
    conv2d_157 (Conv2D)             (None, 5, 5, 384)    884736      activation_156[0][0]             
    __________________________________________________________________________________________________
    conv2d_159 (Conv2D)             (None, 5, 5, 288)    663552      activation_158[0][0]             
    __________________________________________________________________________________________________
    conv2d_162 (Conv2D)             (None, 5, 5, 320)    829440      activation_161[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_157 (BatchN (None, 5, 5, 384)    1152        conv2d_157[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_159 (BatchN (None, 5, 5, 288)    864         conv2d_159[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_162 (BatchN (None, 5, 5, 320)    960         conv2d_162[0][0]                 
    __________________________________________________________________________________________________
    activation_157 (Activation)     (None, 5, 5, 384)    0           batch_normalization_157[0][0]    
    __________________________________________________________________________________________________
    activation_159 (Activation)     (None, 5, 5, 288)    0           batch_normalization_159[0][0]    
    __________________________________________________________________________________________________
    activation_162 (Activation)     (None, 5, 5, 320)    0           batch_normalization_162[0][0]    
    __________________________________________________________________________________________________
    max_pooling2d_3 (MaxPooling2D)  (None, 5, 5, 1088)   0           block17_20_ac[0][0]              
    __________________________________________________________________________________________________
    mixed_7a (Concatenate)          (None, 5, 5, 2080)   0           activation_157[0][0]             
                                                                     activation_159[0][0]             
                                                                     activation_162[0][0]             
                                                                     max_pooling2d_3[0][0]            
    __________________________________________________________________________________________________
    conv2d_164 (Conv2D)             (None, 5, 5, 192)    399360      mixed_7a[0][0]                   
    __________________________________________________________________________________________________
    batch_normalization_164 (BatchN (None, 5, 5, 192)    576         conv2d_164[0][0]                 
    __________________________________________________________________________________________________
    activation_164 (Activation)     (None, 5, 5, 192)    0           batch_normalization_164[0][0]    
    __________________________________________________________________________________________________
    conv2d_165 (Conv2D)             (None, 5, 5, 224)    129024      activation_164[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_165 (BatchN (None, 5, 5, 224)    672         conv2d_165[0][0]                 
    __________________________________________________________________________________________________
    activation_165 (Activation)     (None, 5, 5, 224)    0           batch_normalization_165[0][0]    
    __________________________________________________________________________________________________
    conv2d_163 (Conv2D)             (None, 5, 5, 192)    399360      mixed_7a[0][0]                   
    __________________________________________________________________________________________________
    conv2d_166 (Conv2D)             (None, 5, 5, 256)    172032      activation_165[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_163 (BatchN (None, 5, 5, 192)    576         conv2d_163[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_166 (BatchN (None, 5, 5, 256)    768         conv2d_166[0][0]                 
    __________________________________________________________________________________________________
    activation_163 (Activation)     (None, 5, 5, 192)    0           batch_normalization_163[0][0]    
    __________________________________________________________________________________________________
    activation_166 (Activation)     (None, 5, 5, 256)    0           batch_normalization_166[0][0]    
    __________________________________________________________________________________________________
    block8_1_mixed (Concatenate)    (None, 5, 5, 448)    0           activation_163[0][0]             
                                                                     activation_166[0][0]             
    __________________________________________________________________________________________________
    block8_1_conv (Conv2D)          (None, 5, 5, 2080)   933920      block8_1_mixed[0][0]             
    __________________________________________________________________________________________________
    block8_1 (Lambda)               (None, 5, 5, 2080)   0           mixed_7a[0][0]                   
                                                                     block8_1_conv[0][0]              
    __________________________________________________________________________________________________
    block8_1_ac (Activation)        (None, 5, 5, 2080)   0           block8_1[0][0]                   
    __________________________________________________________________________________________________
    conv2d_168 (Conv2D)             (None, 5, 5, 192)    399360      block8_1_ac[0][0]                
    __________________________________________________________________________________________________
    batch_normalization_168 (BatchN (None, 5, 5, 192)    576         conv2d_168[0][0]                 
    __________________________________________________________________________________________________
    activation_168 (Activation)     (None, 5, 5, 192)    0           batch_normalization_168[0][0]    
    __________________________________________________________________________________________________
    conv2d_169 (Conv2D)             (None, 5, 5, 224)    129024      activation_168[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_169 (BatchN (None, 5, 5, 224)    672         conv2d_169[0][0]                 
    __________________________________________________________________________________________________
    activation_169 (Activation)     (None, 5, 5, 224)    0           batch_normalization_169[0][0]    
    __________________________________________________________________________________________________
    conv2d_167 (Conv2D)             (None, 5, 5, 192)    399360      block8_1_ac[0][0]                
    __________________________________________________________________________________________________
    conv2d_170 (Conv2D)             (None, 5, 5, 256)    172032      activation_169[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_167 (BatchN (None, 5, 5, 192)    576         conv2d_167[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_170 (BatchN (None, 5, 5, 256)    768         conv2d_170[0][0]                 
    __________________________________________________________________________________________________
    activation_167 (Activation)     (None, 5, 5, 192)    0           batch_normalization_167[0][0]    
    __________________________________________________________________________________________________
    activation_170 (Activation)     (None, 5, 5, 256)    0           batch_normalization_170[0][0]    
    __________________________________________________________________________________________________
    block8_2_mixed (Concatenate)    (None, 5, 5, 448)    0           activation_167[0][0]             
                                                                     activation_170[0][0]             
    __________________________________________________________________________________________________
    block8_2_conv (Conv2D)          (None, 5, 5, 2080)   933920      block8_2_mixed[0][0]             
    __________________________________________________________________________________________________
    block8_2 (Lambda)               (None, 5, 5, 2080)   0           block8_1_ac[0][0]                
                                                                     block8_2_conv[0][0]              
    __________________________________________________________________________________________________
    block8_2_ac (Activation)        (None, 5, 5, 2080)   0           block8_2[0][0]                   
    __________________________________________________________________________________________________
    conv2d_172 (Conv2D)             (None, 5, 5, 192)    399360      block8_2_ac[0][0]                
    __________________________________________________________________________________________________
    batch_normalization_172 (BatchN (None, 5, 5, 192)    576         conv2d_172[0][0]                 
    __________________________________________________________________________________________________
    activation_172 (Activation)     (None, 5, 5, 192)    0           batch_normalization_172[0][0]    
    __________________________________________________________________________________________________
    conv2d_173 (Conv2D)             (None, 5, 5, 224)    129024      activation_172[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_173 (BatchN (None, 5, 5, 224)    672         conv2d_173[0][0]                 
    __________________________________________________________________________________________________
    activation_173 (Activation)     (None, 5, 5, 224)    0           batch_normalization_173[0][0]    
    __________________________________________________________________________________________________
    conv2d_171 (Conv2D)             (None, 5, 5, 192)    399360      block8_2_ac[0][0]                
    __________________________________________________________________________________________________
    conv2d_174 (Conv2D)             (None, 5, 5, 256)    172032      activation_173[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_171 (BatchN (None, 5, 5, 192)    576         conv2d_171[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_174 (BatchN (None, 5, 5, 256)    768         conv2d_174[0][0]                 
    __________________________________________________________________________________________________
    activation_171 (Activation)     (None, 5, 5, 192)    0           batch_normalization_171[0][0]    
    __________________________________________________________________________________________________
    activation_174 (Activation)     (None, 5, 5, 256)    0           batch_normalization_174[0][0]    
    __________________________________________________________________________________________________
    block8_3_mixed (Concatenate)    (None, 5, 5, 448)    0           activation_171[0][0]             
                                                                     activation_174[0][0]             
    __________________________________________________________________________________________________
    block8_3_conv (Conv2D)          (None, 5, 5, 2080)   933920      block8_3_mixed[0][0]             
    __________________________________________________________________________________________________
    block8_3 (Lambda)               (None, 5, 5, 2080)   0           block8_2_ac[0][0]                
                                                                     block8_3_conv[0][0]              
    __________________________________________________________________________________________________
    block8_3_ac (Activation)        (None, 5, 5, 2080)   0           block8_3[0][0]                   
    __________________________________________________________________________________________________
    conv2d_176 (Conv2D)             (None, 5, 5, 192)    399360      block8_3_ac[0][0]                
    __________________________________________________________________________________________________
    batch_normalization_176 (BatchN (None, 5, 5, 192)    576         conv2d_176[0][0]                 
    __________________________________________________________________________________________________
    activation_176 (Activation)     (None, 5, 5, 192)    0           batch_normalization_176[0][0]    
    __________________________________________________________________________________________________
    conv2d_177 (Conv2D)             (None, 5, 5, 224)    129024      activation_176[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_177 (BatchN (None, 5, 5, 224)    672         conv2d_177[0][0]                 
    __________________________________________________________________________________________________
    activation_177 (Activation)     (None, 5, 5, 224)    0           batch_normalization_177[0][0]    
    __________________________________________________________________________________________________
    conv2d_175 (Conv2D)             (None, 5, 5, 192)    399360      block8_3_ac[0][0]                
    __________________________________________________________________________________________________
    conv2d_178 (Conv2D)             (None, 5, 5, 256)    172032      activation_177[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_175 (BatchN (None, 5, 5, 192)    576         conv2d_175[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_178 (BatchN (None, 5, 5, 256)    768         conv2d_178[0][0]                 
    __________________________________________________________________________________________________
    activation_175 (Activation)     (None, 5, 5, 192)    0           batch_normalization_175[0][0]    
    __________________________________________________________________________________________________
    activation_178 (Activation)     (None, 5, 5, 256)    0           batch_normalization_178[0][0]    
    __________________________________________________________________________________________________
    block8_4_mixed (Concatenate)    (None, 5, 5, 448)    0           activation_175[0][0]             
                                                                     activation_178[0][0]             
    __________________________________________________________________________________________________
    block8_4_conv (Conv2D)          (None, 5, 5, 2080)   933920      block8_4_mixed[0][0]             
    __________________________________________________________________________________________________
    block8_4 (Lambda)               (None, 5, 5, 2080)   0           block8_3_ac[0][0]                
                                                                     block8_4_conv[0][0]              
    __________________________________________________________________________________________________
    block8_4_ac (Activation)        (None, 5, 5, 2080)   0           block8_4[0][0]                   
    __________________________________________________________________________________________________
    conv2d_180 (Conv2D)             (None, 5, 5, 192)    399360      block8_4_ac[0][0]                
    __________________________________________________________________________________________________
    batch_normalization_180 (BatchN (None, 5, 5, 192)    576         conv2d_180[0][0]                 
    __________________________________________________________________________________________________
    activation_180 (Activation)     (None, 5, 5, 192)    0           batch_normalization_180[0][0]    
    __________________________________________________________________________________________________
    conv2d_181 (Conv2D)             (None, 5, 5, 224)    129024      activation_180[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_181 (BatchN (None, 5, 5, 224)    672         conv2d_181[0][0]                 
    __________________________________________________________________________________________________
    activation_181 (Activation)     (None, 5, 5, 224)    0           batch_normalization_181[0][0]    
    __________________________________________________________________________________________________
    conv2d_179 (Conv2D)             (None, 5, 5, 192)    399360      block8_4_ac[0][0]                
    __________________________________________________________________________________________________
    conv2d_182 (Conv2D)             (None, 5, 5, 256)    172032      activation_181[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_179 (BatchN (None, 5, 5, 192)    576         conv2d_179[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_182 (BatchN (None, 5, 5, 256)    768         conv2d_182[0][0]                 
    __________________________________________________________________________________________________
    activation_179 (Activation)     (None, 5, 5, 192)    0           batch_normalization_179[0][0]    
    __________________________________________________________________________________________________
    activation_182 (Activation)     (None, 5, 5, 256)    0           batch_normalization_182[0][0]    
    __________________________________________________________________________________________________
    block8_5_mixed (Concatenate)    (None, 5, 5, 448)    0           activation_179[0][0]             
                                                                     activation_182[0][0]             
    __________________________________________________________________________________________________
    block8_5_conv (Conv2D)          (None, 5, 5, 2080)   933920      block8_5_mixed[0][0]             
    __________________________________________________________________________________________________
    block8_5 (Lambda)               (None, 5, 5, 2080)   0           block8_4_ac[0][0]                
                                                                     block8_5_conv[0][0]              
    __________________________________________________________________________________________________
    block8_5_ac (Activation)        (None, 5, 5, 2080)   0           block8_5[0][0]                   
    __________________________________________________________________________________________________
    conv2d_184 (Conv2D)             (None, 5, 5, 192)    399360      block8_5_ac[0][0]                
    __________________________________________________________________________________________________
    batch_normalization_184 (BatchN (None, 5, 5, 192)    576         conv2d_184[0][0]                 
    __________________________________________________________________________________________________
    activation_184 (Activation)     (None, 5, 5, 192)    0           batch_normalization_184[0][0]    
    __________________________________________________________________________________________________
    conv2d_185 (Conv2D)             (None, 5, 5, 224)    129024      activation_184[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_185 (BatchN (None, 5, 5, 224)    672         conv2d_185[0][0]                 
    __________________________________________________________________________________________________
    activation_185 (Activation)     (None, 5, 5, 224)    0           batch_normalization_185[0][0]    
    __________________________________________________________________________________________________
    conv2d_183 (Conv2D)             (None, 5, 5, 192)    399360      block8_5_ac[0][0]                
    __________________________________________________________________________________________________
    conv2d_186 (Conv2D)             (None, 5, 5, 256)    172032      activation_185[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_183 (BatchN (None, 5, 5, 192)    576         conv2d_183[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_186 (BatchN (None, 5, 5, 256)    768         conv2d_186[0][0]                 
    __________________________________________________________________________________________________
    activation_183 (Activation)     (None, 5, 5, 192)    0           batch_normalization_183[0][0]    
    __________________________________________________________________________________________________
    activation_186 (Activation)     (None, 5, 5, 256)    0           batch_normalization_186[0][0]    
    __________________________________________________________________________________________________
    block8_6_mixed (Concatenate)    (None, 5, 5, 448)    0           activation_183[0][0]             
                                                                     activation_186[0][0]             
    __________________________________________________________________________________________________
    block8_6_conv (Conv2D)          (None, 5, 5, 2080)   933920      block8_6_mixed[0][0]             
    __________________________________________________________________________________________________
    block8_6 (Lambda)               (None, 5, 5, 2080)   0           block8_5_ac[0][0]                
                                                                     block8_6_conv[0][0]              
    __________________________________________________________________________________________________
    block8_6_ac (Activation)        (None, 5, 5, 2080)   0           block8_6[0][0]                   
    __________________________________________________________________________________________________
    conv2d_188 (Conv2D)             (None, 5, 5, 192)    399360      block8_6_ac[0][0]                
    __________________________________________________________________________________________________
    batch_normalization_188 (BatchN (None, 5, 5, 192)    576         conv2d_188[0][0]                 
    __________________________________________________________________________________________________
    activation_188 (Activation)     (None, 5, 5, 192)    0           batch_normalization_188[0][0]    
    __________________________________________________________________________________________________
    conv2d_189 (Conv2D)             (None, 5, 5, 224)    129024      activation_188[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_189 (BatchN (None, 5, 5, 224)    672         conv2d_189[0][0]                 
    __________________________________________________________________________________________________
    activation_189 (Activation)     (None, 5, 5, 224)    0           batch_normalization_189[0][0]    
    __________________________________________________________________________________________________
    conv2d_187 (Conv2D)             (None, 5, 5, 192)    399360      block8_6_ac[0][0]                
    __________________________________________________________________________________________________
    conv2d_190 (Conv2D)             (None, 5, 5, 256)    172032      activation_189[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_187 (BatchN (None, 5, 5, 192)    576         conv2d_187[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_190 (BatchN (None, 5, 5, 256)    768         conv2d_190[0][0]                 
    __________________________________________________________________________________________________
    activation_187 (Activation)     (None, 5, 5, 192)    0           batch_normalization_187[0][0]    
    __________________________________________________________________________________________________
    activation_190 (Activation)     (None, 5, 5, 256)    0           batch_normalization_190[0][0]    
    __________________________________________________________________________________________________
    block8_7_mixed (Concatenate)    (None, 5, 5, 448)    0           activation_187[0][0]             
                                                                     activation_190[0][0]             
    __________________________________________________________________________________________________
    block8_7_conv (Conv2D)          (None, 5, 5, 2080)   933920      block8_7_mixed[0][0]             
    __________________________________________________________________________________________________
    block8_7 (Lambda)               (None, 5, 5, 2080)   0           block8_6_ac[0][0]                
                                                                     block8_7_conv[0][0]              
    __________________________________________________________________________________________________
    block8_7_ac (Activation)        (None, 5, 5, 2080)   0           block8_7[0][0]                   
    __________________________________________________________________________________________________
    conv2d_192 (Conv2D)             (None, 5, 5, 192)    399360      block8_7_ac[0][0]                
    __________________________________________________________________________________________________
    batch_normalization_192 (BatchN (None, 5, 5, 192)    576         conv2d_192[0][0]                 
    __________________________________________________________________________________________________
    activation_192 (Activation)     (None, 5, 5, 192)    0           batch_normalization_192[0][0]    
    __________________________________________________________________________________________________
    conv2d_193 (Conv2D)             (None, 5, 5, 224)    129024      activation_192[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_193 (BatchN (None, 5, 5, 224)    672         conv2d_193[0][0]                 
    __________________________________________________________________________________________________
    activation_193 (Activation)     (None, 5, 5, 224)    0           batch_normalization_193[0][0]    
    __________________________________________________________________________________________________
    conv2d_191 (Conv2D)             (None, 5, 5, 192)    399360      block8_7_ac[0][0]                
    __________________________________________________________________________________________________
    conv2d_194 (Conv2D)             (None, 5, 5, 256)    172032      activation_193[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_191 (BatchN (None, 5, 5, 192)    576         conv2d_191[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_194 (BatchN (None, 5, 5, 256)    768         conv2d_194[0][0]                 
    __________________________________________________________________________________________________
    activation_191 (Activation)     (None, 5, 5, 192)    0           batch_normalization_191[0][0]    
    __________________________________________________________________________________________________
    activation_194 (Activation)     (None, 5, 5, 256)    0           batch_normalization_194[0][0]    
    __________________________________________________________________________________________________
    block8_8_mixed (Concatenate)    (None, 5, 5, 448)    0           activation_191[0][0]             
                                                                     activation_194[0][0]             
    __________________________________________________________________________________________________
    block8_8_conv (Conv2D)          (None, 5, 5, 2080)   933920      block8_8_mixed[0][0]             
    __________________________________________________________________________________________________
    block8_8 (Lambda)               (None, 5, 5, 2080)   0           block8_7_ac[0][0]                
                                                                     block8_8_conv[0][0]              
    __________________________________________________________________________________________________
    block8_8_ac (Activation)        (None, 5, 5, 2080)   0           block8_8[0][0]                   
    __________________________________________________________________________________________________
    conv2d_196 (Conv2D)             (None, 5, 5, 192)    399360      block8_8_ac[0][0]                
    __________________________________________________________________________________________________
    batch_normalization_196 (BatchN (None, 5, 5, 192)    576         conv2d_196[0][0]                 
    __________________________________________________________________________________________________
    activation_196 (Activation)     (None, 5, 5, 192)    0           batch_normalization_196[0][0]    
    __________________________________________________________________________________________________
    conv2d_197 (Conv2D)             (None, 5, 5, 224)    129024      activation_196[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_197 (BatchN (None, 5, 5, 224)    672         conv2d_197[0][0]                 
    __________________________________________________________________________________________________
    activation_197 (Activation)     (None, 5, 5, 224)    0           batch_normalization_197[0][0]    
    __________________________________________________________________________________________________
    conv2d_195 (Conv2D)             (None, 5, 5, 192)    399360      block8_8_ac[0][0]                
    __________________________________________________________________________________________________
    conv2d_198 (Conv2D)             (None, 5, 5, 256)    172032      activation_197[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_195 (BatchN (None, 5, 5, 192)    576         conv2d_195[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_198 (BatchN (None, 5, 5, 256)    768         conv2d_198[0][0]                 
    __________________________________________________________________________________________________
    activation_195 (Activation)     (None, 5, 5, 192)    0           batch_normalization_195[0][0]    
    __________________________________________________________________________________________________
    activation_198 (Activation)     (None, 5, 5, 256)    0           batch_normalization_198[0][0]    
    __________________________________________________________________________________________________
    block8_9_mixed (Concatenate)    (None, 5, 5, 448)    0           activation_195[0][0]             
                                                                     activation_198[0][0]             
    __________________________________________________________________________________________________
    block8_9_conv (Conv2D)          (None, 5, 5, 2080)   933920      block8_9_mixed[0][0]             
    __________________________________________________________________________________________________
    block8_9 (Lambda)               (None, 5, 5, 2080)   0           block8_8_ac[0][0]                
                                                                     block8_9_conv[0][0]              
    __________________________________________________________________________________________________
    block8_9_ac (Activation)        (None, 5, 5, 2080)   0           block8_9[0][0]                   
    __________________________________________________________________________________________________
    conv2d_200 (Conv2D)             (None, 5, 5, 192)    399360      block8_9_ac[0][0]                
    __________________________________________________________________________________________________
    batch_normalization_200 (BatchN (None, 5, 5, 192)    576         conv2d_200[0][0]                 
    __________________________________________________________________________________________________
    activation_200 (Activation)     (None, 5, 5, 192)    0           batch_normalization_200[0][0]    
    __________________________________________________________________________________________________
    conv2d_201 (Conv2D)             (None, 5, 5, 224)    129024      activation_200[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_201 (BatchN (None, 5, 5, 224)    672         conv2d_201[0][0]                 
    __________________________________________________________________________________________________
    activation_201 (Activation)     (None, 5, 5, 224)    0           batch_normalization_201[0][0]    
    __________________________________________________________________________________________________
    conv2d_199 (Conv2D)             (None, 5, 5, 192)    399360      block8_9_ac[0][0]                
    __________________________________________________________________________________________________
    conv2d_202 (Conv2D)             (None, 5, 5, 256)    172032      activation_201[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_199 (BatchN (None, 5, 5, 192)    576         conv2d_199[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_202 (BatchN (None, 5, 5, 256)    768         conv2d_202[0][0]                 
    __________________________________________________________________________________________________
    activation_199 (Activation)     (None, 5, 5, 192)    0           batch_normalization_199[0][0]    
    __________________________________________________________________________________________________
    activation_202 (Activation)     (None, 5, 5, 256)    0           batch_normalization_202[0][0]    
    __________________________________________________________________________________________________
    block8_10_mixed (Concatenate)   (None, 5, 5, 448)    0           activation_199[0][0]             
                                                                     activation_202[0][0]             
    __________________________________________________________________________________________________
    block8_10_conv (Conv2D)         (None, 5, 5, 2080)   933920      block8_10_mixed[0][0]            
    __________________________________________________________________________________________________
    block8_10 (Lambda)              (None, 5, 5, 2080)   0           block8_9_ac[0][0]                
                                                                     block8_10_conv[0][0]             
    __________________________________________________________________________________________________
    conv_7b (Conv2D)                (None, 5, 5, 1536)   3194880     block8_10[0][0]                  
    __________________________________________________________________________________________________
    conv_7b_bn (BatchNormalization) (None, 5, 5, 1536)   4608        conv_7b[0][0]                    
    __________________________________________________________________________________________________
    conv_7b_ac (Activation)         (None, 5, 5, 1536)   0           conv_7b_bn[0][0]                 
    __________________________________________________________________________________________________
    flatten_1 (Flatten)             (None, 38400)        0           conv_7b_ac[0][0]                 
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 4096)         157290496   flatten_1[0][0]                  
    __________________________________________________________________________________________________
    re_lu_2 (ReLU)                  (None, 4096)         0           dense_3[0][0]                    
    __________________________________________________________________________________________________
    dropout_2 (Dropout)             (None, 4096)         0           re_lu_2[0][0]                    
    __________________________________________________________________________________________________
    dense_4 (Dense)                 (None, 4096)         16781312    dropout_2[0][0]                  
    __________________________________________________________________________________________________
    re_lu_3 (ReLU)                  (None, 4096)         0           dense_4[0][0]                    
    __________________________________________________________________________________________________
    dropout_3 (Dropout)             (None, 4096)         0           re_lu_3[0][0]                    
    __________________________________________________________________________________________________
    dense_5 (Dense)                 (None, 3)            12291       dropout_3[0][0]                  
    ==================================================================================================
    Total params: 228,420,835
    Trainable params: 177,280,515
    Non-trainable params: 51,140,320
    __________________________________________________________________________________________________
    None
    Epoch 1/50
    142/142 [==============================] - 53s 291ms/step - loss: 1.0784 - accuracy: 0.6410 - auc: 0.9284 - cohen_kappa: 0.6704 - f1_score: 0.7758 - precision: 0.8024 - recall: 0.7810 - val_loss: 0.2912 - val_accuracy: 0.8990 - val_auc: 0.9808 - val_cohen_kappa: 0.8478 - val_f1_score: 0.8957 - val_precision: 0.9044 - val_recall: 0.8969
    Epoch 2/50
    142/142 [==============================] - 37s 264ms/step - loss: 0.3821 - accuracy: 0.8938 - auc: 0.9711 - cohen_kappa: 0.8352 - f1_score: 0.8865 - precision: 0.8945 - recall: 0.8862 - val_loss: 0.0507 - val_accuracy: 0.9773 - val_auc: 0.9994 - val_cohen_kappa: 0.9652 - val_f1_score: 0.9777 - val_precision: 0.9814 - val_recall: 0.9773
    Epoch 3/50
    142/142 [==============================] - 39s 274ms/step - loss: 0.2045 - accuracy: 0.9362 - auc: 0.9881 - cohen_kappa: 0.9012 - f1_score: 0.9325 - precision: 0.9371 - recall: 0.9362 - val_loss: 0.1504 - val_accuracy: 0.9443 - val_auc: 0.9912 - val_cohen_kappa: 0.9155 - val_f1_score: 0.9406 - val_precision: 0.9443 - val_recall: 0.9443
    Epoch 4/50
    142/142 [==============================] - 38s 271ms/step - loss: 0.1144 - accuracy: 0.9632 - auc: 0.9950 - cohen_kappa: 0.9427 - f1_score: 0.9623 - precision: 0.9642 - recall: 0.9630 - val_loss: 0.0348 - val_accuracy: 0.9938 - val_auc: 0.9982 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9934 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 5/50
    142/142 [==============================] - 38s 268ms/step - loss: 0.1253 - accuracy: 0.9675 - auc: 0.9936 - cohen_kappa: 0.9491 - f1_score: 0.9662 - precision: 0.9677 - recall: 0.9675 - val_loss: 0.0397 - val_accuracy: 0.9835 - val_auc: 0.9995 - val_cohen_kappa: 0.9747 - val_f1_score: 0.9842 - val_precision: 0.9835 - val_recall: 0.9835
    Epoch 6/50
    142/142 [==============================] - 38s 266ms/step - loss: 0.1198 - accuracy: 0.9642 - auc: 0.9941 - cohen_kappa: 0.9440 - f1_score: 0.9633 - precision: 0.9664 - recall: 0.9624 - val_loss: 0.0305 - val_accuracy: 0.9938 - val_auc: 0.9982 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9940 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 7/50
    142/142 [==============================] - 37s 262ms/step - loss: 0.0873 - accuracy: 0.9720 - auc: 0.9970 - cohen_kappa: 0.9563 - f1_score: 0.9725 - precision: 0.9724 - recall: 0.9691 - val_loss: 0.0690 - val_accuracy: 0.9753 - val_auc: 0.9975 - val_cohen_kappa: 0.9622 - val_f1_score: 0.9730 - val_precision: 0.9753 - val_recall: 0.9753
    Epoch 8/50
    142/142 [==============================] - 37s 263ms/step - loss: 0.0887 - accuracy: 0.9736 - auc: 0.9964 - cohen_kappa: 0.9586 - f1_score: 0.9727 - precision: 0.9741 - recall: 0.9735 - val_loss: 0.0400 - val_accuracy: 0.9856 - val_auc: 0.9996 - val_cohen_kappa: 0.9779 - val_f1_score: 0.9844 - val_precision: 0.9856 - val_recall: 0.9856
    Epoch 9/50
    142/142 [==============================] - 39s 271ms/step - loss: 0.0385 - accuracy: 0.9863 - auc: 0.9994 - cohen_kappa: 0.9785 - f1_score: 0.9857 - precision: 0.9862 - recall: 0.9856 - val_loss: 0.0186 - val_accuracy: 0.9897 - val_auc: 0.9999 - val_cohen_kappa: 0.9842 - val_f1_score: 0.9897 - val_precision: 0.9897 - val_recall: 0.9897
    Epoch 10/50
    142/142 [==============================] - 38s 264ms/step - loss: 0.0353 - accuracy: 0.9837 - auc: 0.9996 - cohen_kappa: 0.9746 - f1_score: 0.9837 - precision: 0.9837 - recall: 0.9828 - val_loss: 0.0104 - val_accuracy: 1.0000 - val_auc: 1.0000 - val_cohen_kappa: 1.0000 - val_f1_score: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
    Epoch 11/50
    142/142 [==============================] - 38s 269ms/step - loss: 0.0624 - accuracy: 0.9816 - auc: 0.9970 - cohen_kappa: 0.9710 - f1_score: 0.9773 - precision: 0.9816 - recall: 0.9816 - val_loss: 0.0297 - val_accuracy: 0.9918 - val_auc: 0.9983 - val_cohen_kappa: 0.9874 - val_f1_score: 0.9910 - val_precision: 0.9918 - val_recall: 0.9918
    Epoch 12/50
    142/142 [==============================] - 37s 263ms/step - loss: 0.0441 - accuracy: 0.9876 - auc: 0.9985 - cohen_kappa: 0.9806 - f1_score: 0.9863 - precision: 0.9878 - recall: 0.9869 - val_loss: 0.0328 - val_accuracy: 0.9938 - val_auc: 0.9983 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9945 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 13/50
    142/142 [==============================] - 38s 264ms/step - loss: 0.0243 - accuracy: 0.9903 - auc: 0.9995 - cohen_kappa: 0.9845 - f1_score: 0.9882 - precision: 0.9907 - recall: 0.9903 - val_loss: 0.0287 - val_accuracy: 0.9938 - val_auc: 0.9983 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9934 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 14/50
    142/142 [==============================] - 38s 268ms/step - loss: 0.0371 - accuracy: 0.9853 - auc: 0.9996 - cohen_kappa: 0.9767 - f1_score: 0.9840 - precision: 0.9873 - recall: 0.9853 - val_loss: 0.0410 - val_accuracy: 0.9918 - val_auc: 0.9983 - val_cohen_kappa: 0.9874 - val_f1_score: 0.9915 - val_precision: 0.9918 - val_recall: 0.9918
    Epoch 15/50
    142/142 [==============================] - 38s 269ms/step - loss: 0.0366 - accuracy: 0.9911 - auc: 0.9981 - cohen_kappa: 0.9859 - f1_score: 0.9905 - precision: 0.9911 - recall: 0.9904 - val_loss: 0.0515 - val_accuracy: 0.9897 - val_auc: 0.9982 - val_cohen_kappa: 0.9842 - val_f1_score: 0.9897 - val_precision: 0.9897 - val_recall: 0.9876
    Epoch 16/50
    142/142 [==============================] - 38s 267ms/step - loss: 0.0281 - accuracy: 0.9913 - auc: 0.9987 - cohen_kappa: 0.9864 - f1_score: 0.9913 - precision: 0.9927 - recall: 0.9913 - val_loss: 0.0370 - val_accuracy: 0.9918 - val_auc: 0.9983 - val_cohen_kappa: 0.9874 - val_f1_score: 0.9915 - val_precision: 0.9918 - val_recall: 0.9918
    Epoch 17/50
    142/142 [==============================] - 38s 271ms/step - loss: 0.0184 - accuracy: 0.9931 - auc: 0.9999 - cohen_kappa: 0.9892 - f1_score: 0.9928 - precision: 0.9931 - recall: 0.9931 - val_loss: 0.0289 - val_accuracy: 0.9938 - val_auc: 0.9983 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9934 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 18/50
    142/142 [==============================] - 38s 267ms/step - loss: 0.0115 - accuracy: 0.9954 - auc: 0.9999 - cohen_kappa: 0.9929 - f1_score: 0.9956 - precision: 0.9954 - recall: 0.9954 - val_loss: 0.0161 - val_accuracy: 0.9938 - val_auc: 0.9999 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9934 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 19/50
    142/142 [==============================] - 38s 267ms/step - loss: 0.0090 - accuracy: 0.9963 - auc: 1.0000 - cohen_kappa: 0.9942 - f1_score: 0.9962 - precision: 0.9967 - recall: 0.9963 - val_loss: 0.0243 - val_accuracy: 0.9938 - val_auc: 0.9999 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9934 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 20/50
    142/142 [==============================] - 38s 267ms/step - loss: 0.0324 - accuracy: 0.9952 - auc: 0.9980 - cohen_kappa: 0.9925 - f1_score: 0.9950 - precision: 0.9952 - recall: 0.9952 - val_loss: 0.0241 - val_accuracy: 0.9938 - val_auc: 0.9999 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9934 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 21/50
    142/142 [==============================] - 39s 274ms/step - loss: 0.0103 - accuracy: 0.9970 - auc: 1.0000 - cohen_kappa: 0.9953 - f1_score: 0.9968 - precision: 0.9970 - recall: 0.9970 - val_loss: 0.0240 - val_accuracy: 0.9938 - val_auc: 0.9999 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9934 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 22/50
    142/142 [==============================] - 38s 268ms/step - loss: 0.0068 - accuracy: 0.9971 - auc: 1.0000 - cohen_kappa: 0.9955 - f1_score: 0.9968 - precision: 0.9974 - recall: 0.9971 - val_loss: 0.0242 - val_accuracy: 0.9938 - val_auc: 0.9999 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9934 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 23/50
    142/142 [==============================] - 38s 265ms/step - loss: 0.0098 - accuracy: 0.9962 - auc: 1.0000 - cohen_kappa: 0.9941 - f1_score: 0.9956 - precision: 0.9962 - recall: 0.9962 - val_loss: 0.0238 - val_accuracy: 0.9938 - val_auc: 0.9999 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9934 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 24/50
    142/142 [==============================] - 39s 274ms/step - loss: 0.0169 - accuracy: 0.9952 - auc: 0.9998 - cohen_kappa: 0.9925 - f1_score: 0.9952 - precision: 0.9962 - recall: 0.9952 - val_loss: 0.0234 - val_accuracy: 0.9938 - val_auc: 0.9999 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9934 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 25/50
    142/142 [==============================] - 38s 266ms/step - loss: 0.0055 - accuracy: 0.9986 - auc: 1.0000 - cohen_kappa: 0.9978 - f1_score: 0.9988 - precision: 0.9986 - recall: 0.9986 - val_loss: 0.0226 - val_accuracy: 0.9938 - val_auc: 0.9999 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9934 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 00025: early stopping
    


```python
evaluate_model(incres_model, incres_history, test_generator)
```

    
    Test set accuracy: 1.0 
    
    31/31 [==============================] - 5s 80ms/step
    
                  precision    recall  f1-score   support
    
             AMD       1.00      1.00      1.00       121
             DME       1.00      1.00      1.00       155
          NORMAL       1.00      1.00      1.00       209
    
        accuracy                           1.00       485
       macro avg       1.00      1.00      1.00       485
    weighted avg       1.00      1.00      1.00       485
    
    


![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_27_1.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_27_2.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_27_3.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_27_4.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_27_5.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_27_6.png)


    ROC AUC score: 1.0
    


```python
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


```python
model = custom_model()
```


```python
model, history = train_model(model, train_generator, val_generator, 100, tf.keras.optimizers.SGD(lr=0.001, momentum=0.9), metrics)
```

    Model: "model_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_3 (InputLayer)         [(None, 224, 224, 3)]     0         
    _________________________________________________________________
    conv2d_203 (Conv2D)          (None, 222, 222, 64)      1792      
    _________________________________________________________________
    conv2d_204 (Conv2D)          (None, 220, 220, 64)      36928     
    _________________________________________________________________
    conv2d_205 (Conv2D)          (None, 218, 218, 64)      36928     
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 109, 109, 64)      0         
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 109, 109, 64)      0         
    _________________________________________________________________
    conv2d_206 (Conv2D)          (None, 107, 107, 128)     73856     
    _________________________________________________________________
    conv2d_207 (Conv2D)          (None, 105, 105, 128)     147584    
    _________________________________________________________________
    conv2d_208 (Conv2D)          (None, 103, 103, 128)     147584    
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 51, 51, 128)       0         
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 51, 51, 128)       0         
    _________________________________________________________________
    conv2d_209 (Conv2D)          (None, 49, 49, 256)       295168    
    _________________________________________________________________
    conv2d_210 (Conv2D)          (None, 47, 47, 256)       590080    
    _________________________________________________________________
    conv2d_211 (Conv2D)          (None, 45, 45, 256)       590080    
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 22, 22, 256)       0         
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 22, 22, 256)       0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 123904)            0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 4096)              507514880 
    _________________________________________________________________
    re_lu_4 (ReLU)               (None, 4096)              0         
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 4096)              0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 4096)              16781312  
    _________________________________________________________________
    re_lu_5 (ReLU)               (None, 4096)              0         
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 4096)              0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 3)                 12291     
    =================================================================
    Total params: 526,228,483
    Trainable params: 526,228,483
    Non-trainable params: 0
    _________________________________________________________________
    None
    Epoch 1/100
    142/142 [==============================] - 47s 301ms/step - loss: 1.0770 - accuracy: 0.3923 - auc: 0.8243 - cohen_kappa: 0.3819 - f1_score: 0.5871 - precision: 0.9594 - recall: 0.3840 - val_loss: 1.0521 - val_accuracy: 0.4206 - val_auc: 0.6478 - val_cohen_kappa: 0.0283 - val_f1_score: 0.2232 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/100
    142/142 [==============================] - 41s 289ms/step - loss: 0.9745 - accuracy: 0.5317 - auc: 0.7138 - cohen_kappa: 0.1673 - f1_score: 0.3741 - precision: 0.6119 - recall: 0.3233 - val_loss: 0.9526 - val_accuracy: 0.5588 - val_auc: 0.7398 - val_cohen_kappa: 0.2867 - val_f1_score: 0.4833 - val_precision: 0.6055 - val_recall: 0.2722
    Epoch 3/100
    142/142 [==============================] - 41s 289ms/step - loss: 0.9110 - accuracy: 0.5738 - auc: 0.7578 - cohen_kappa: 0.2924 - f1_score: 0.5173 - precision: 0.6180 - recall: 0.4194 - val_loss: 0.9805 - val_accuracy: 0.5649 - val_auc: 0.7674 - val_cohen_kappa: 0.3039 - val_f1_score: 0.5274 - val_precision: 1.0000 - val_recall: 0.0021
    Epoch 4/100
    142/142 [==============================] - 43s 303ms/step - loss: 0.9514 - accuracy: 0.5313 - auc: 0.7301 - cohen_kappa: 0.2432 - f1_score: 0.5000 - precision: 0.5719 - recall: 0.3206 - val_loss: 1.0089 - val_accuracy: 0.5402 - val_auc: 0.7487 - val_cohen_kappa: 0.2849 - val_f1_score: 0.5176 - val_precision: 1.0000 - val_recall: 0.0165
    Epoch 5/100
    142/142 [==============================] - 41s 289ms/step - loss: 0.9181 - accuracy: 0.5966 - auc: 0.7618 - cohen_kappa: 0.3390 - f1_score: 0.5414 - precision: 0.6294 - recall: 0.3856 - val_loss: 0.8970 - val_accuracy: 0.5876 - val_auc: 0.7698 - val_cohen_kappa: 0.3336 - val_f1_score: 0.5220 - val_precision: 0.5966 - val_recall: 0.5093
    Epoch 6/100
    142/142 [==============================] - 41s 286ms/step - loss: 0.8468 - accuracy: 0.6118 - auc: 0.7946 - cohen_kappa: 0.3540 - f1_score: 0.5600 - precision: 0.6450 - recall: 0.5178 - val_loss: 0.8816 - val_accuracy: 0.5794 - val_auc: 0.7910 - val_cohen_kappa: 0.3232 - val_f1_score: 0.5246 - val_precision: 0.6955 - val_recall: 0.4144
    Epoch 7/100
    142/142 [==============================] - 42s 293ms/step - loss: 0.8544 - accuracy: 0.6195 - auc: 0.7911 - cohen_kappa: 0.3787 - f1_score: 0.5895 - precision: 0.6618 - recall: 0.4868 - val_loss: 0.9469 - val_accuracy: 0.5918 - val_auc: 0.7554 - val_cohen_kappa: 0.3692 - val_f1_score: 0.5705 - val_precision: 0.8333 - val_recall: 0.1031
    Epoch 8/100
    142/142 [==============================] - 41s 290ms/step - loss: 0.8921 - accuracy: 0.5903 - auc: 0.7682 - cohen_kappa: 0.3360 - f1_score: 0.5505 - precision: 0.6204 - recall: 0.4319 - val_loss: 0.9378 - val_accuracy: 0.6186 - val_auc: 0.7736 - val_cohen_kappa: 0.4099 - val_f1_score: 0.6063 - val_precision: 0.8929 - val_recall: 0.1031
    Epoch 9/100
    142/142 [==============================] - 42s 297ms/step - loss: 0.8525 - accuracy: 0.6246 - auc: 0.7897 - cohen_kappa: 0.3800 - f1_score: 0.5806 - precision: 0.6503 - recall: 0.5157 - val_loss: 0.8720 - val_accuracy: 0.6082 - val_auc: 0.8159 - val_cohen_kappa: 0.3820 - val_f1_score: 0.5827 - val_precision: 0.8197 - val_recall: 0.3093
    Epoch 10/100
    142/142 [==============================] - 41s 290ms/step - loss: 0.8334 - accuracy: 0.6366 - auc: 0.8009 - cohen_kappa: 0.4052 - f1_score: 0.5976 - precision: 0.6786 - recall: 0.5348 - val_loss: 0.8685 - val_accuracy: 0.6268 - val_auc: 0.8300 - val_cohen_kappa: 0.4170 - val_f1_score: 0.6015 - val_precision: 0.8208 - val_recall: 0.2928
    Epoch 11/100
    142/142 [==============================] - 42s 293ms/step - loss: 0.8432 - accuracy: 0.5947 - auc: 0.7940 - cohen_kappa: 0.3533 - f1_score: 0.5713 - precision: 0.6492 - recall: 0.4935 - val_loss: 0.8618 - val_accuracy: 0.6000 - val_auc: 0.8190 - val_cohen_kappa: 0.3589 - val_f1_score: 0.5480 - val_precision: 0.7614 - val_recall: 0.4474
    Epoch 12/100
    142/142 [==============================] - 41s 290ms/step - loss: 0.7939 - accuracy: 0.6505 - auc: 0.8262 - cohen_kappa: 0.4244 - f1_score: 0.6204 - precision: 0.6958 - recall: 0.5588 - val_loss: 0.8200 - val_accuracy: 0.5773 - val_auc: 0.8113 - val_cohen_kappa: 0.3168 - val_f1_score: 0.5037 - val_precision: 0.6131 - val_recall: 0.5588
    Epoch 13/100
    142/142 [==============================] - 42s 298ms/step - loss: 0.7944 - accuracy: 0.6405 - auc: 0.8222 - cohen_kappa: 0.4251 - f1_score: 0.6111 - precision: 0.6819 - recall: 0.5628 - val_loss: 1.0746 - val_accuracy: 0.4000 - val_auc: 0.5999 - val_cohen_kappa: 0.1623 - val_f1_score: 0.3730 - val_precision: 0.5202 - val_recall: 0.1856
    Epoch 14/100
    142/142 [==============================] - 41s 288ms/step - loss: 0.8151 - accuracy: 0.6393 - auc: 0.8111 - cohen_kappa: 0.4106 - f1_score: 0.5970 - precision: 0.6891 - recall: 0.5291 - val_loss: 0.7747 - val_accuracy: 0.6392 - val_auc: 0.8456 - val_cohen_kappa: 0.4284 - val_f1_score: 0.6106 - val_precision: 0.7542 - val_recall: 0.5567
    Epoch 15/100
    142/142 [==============================] - 42s 294ms/step - loss: 0.7827 - accuracy: 0.6451 - auc: 0.8238 - cohen_kappa: 0.4365 - f1_score: 0.6228 - precision: 0.6773 - recall: 0.5592 - val_loss: 0.8799 - val_accuracy: 0.6247 - val_auc: 0.8025 - val_cohen_kappa: 0.4428 - val_f1_score: 0.6209 - val_precision: 0.7330 - val_recall: 0.3113
    Epoch 16/100
    142/142 [==============================] - 41s 289ms/step - loss: 0.7910 - accuracy: 0.6551 - auc: 0.8225 - cohen_kappa: 0.4402 - f1_score: 0.6207 - precision: 0.6891 - recall: 0.5632 - val_loss: 0.7327 - val_accuracy: 0.6474 - val_auc: 0.8585 - val_cohen_kappa: 0.4420 - val_f1_score: 0.6155 - val_precision: 0.7206 - val_recall: 0.5691
    Epoch 17/100
    142/142 [==============================] - 41s 291ms/step - loss: 0.7629 - accuracy: 0.6665 - auc: 0.8368 - cohen_kappa: 0.4607 - f1_score: 0.6271 - precision: 0.7012 - recall: 0.5871 - val_loss: 0.7672 - val_accuracy: 0.6639 - val_auc: 0.8596 - val_cohen_kappa: 0.4738 - val_f1_score: 0.6347 - val_precision: 0.7720 - val_recall: 0.4887
    Epoch 18/100
    142/142 [==============================] - 43s 305ms/step - loss: 0.7116 - accuracy: 0.6836 - auc: 0.8571 - cohen_kappa: 0.4938 - f1_score: 0.6516 - precision: 0.7211 - recall: 0.6187 - val_loss: 0.7134 - val_accuracy: 0.6887 - val_auc: 0.8707 - val_cohen_kappa: 0.5111 - val_f1_score: 0.6653 - val_precision: 0.7678 - val_recall: 0.6000
    Epoch 19/100
    142/142 [==============================] - 42s 291ms/step - loss: 0.7182 - accuracy: 0.6708 - auc: 0.8535 - cohen_kappa: 0.4690 - f1_score: 0.6389 - precision: 0.7054 - recall: 0.5968 - val_loss: 0.7045 - val_accuracy: 0.6969 - val_auc: 0.8811 - val_cohen_kappa: 0.5281 - val_f1_score: 0.6692 - val_precision: 0.7642 - val_recall: 0.5814
    Epoch 20/100
    142/142 [==============================] - 42s 293ms/step - loss: 0.6879 - accuracy: 0.6910 - auc: 0.8664 - cohen_kappa: 0.5033 - f1_score: 0.6666 - precision: 0.7179 - recall: 0.6381 - val_loss: 0.7558 - val_accuracy: 0.6577 - val_auc: 0.8441 - val_cohen_kappa: 0.4556 - val_f1_score: 0.6256 - val_precision: 0.6934 - val_recall: 0.6247
    Epoch 21/100
    142/142 [==============================] - 41s 292ms/step - loss: 0.7142 - accuracy: 0.6775 - auc: 0.8558 - cohen_kappa: 0.4915 - f1_score: 0.6578 - precision: 0.7118 - recall: 0.6076 - val_loss: 0.7113 - val_accuracy: 0.6784 - val_auc: 0.8604 - val_cohen_kappa: 0.4866 - val_f1_score: 0.6468 - val_precision: 0.6975 - val_recall: 0.6227
    Epoch 22/100
    142/142 [==============================] - 42s 293ms/step - loss: 0.6481 - accuracy: 0.7339 - auc: 0.8834 - cohen_kappa: 0.5700 - f1_score: 0.7047 - precision: 0.7630 - recall: 0.6813 - val_loss: 0.6290 - val_accuracy: 0.7278 - val_auc: 0.8973 - val_cohen_kappa: 0.5775 - val_f1_score: 0.7101 - val_precision: 0.7720 - val_recall: 0.6701
    Epoch 23/100
    142/142 [==============================] - 43s 302ms/step - loss: 0.6296 - accuracy: 0.7240 - auc: 0.8891 - cohen_kappa: 0.5629 - f1_score: 0.7080 - precision: 0.7512 - recall: 0.6820 - val_loss: 0.7384 - val_accuracy: 0.6289 - val_auc: 0.8497 - val_cohen_kappa: 0.4051 - val_f1_score: 0.5884 - val_precision: 0.6783 - val_recall: 0.6000
    Epoch 24/100
    142/142 [==============================] - 42s 291ms/step - loss: 0.6685 - accuracy: 0.7068 - auc: 0.8744 - cohen_kappa: 0.5287 - f1_score: 0.6787 - precision: 0.7397 - recall: 0.6556 - val_loss: 0.6914 - val_accuracy: 0.6825 - val_auc: 0.8834 - val_cohen_kappa: 0.4936 - val_f1_score: 0.6555 - val_precision: 0.7769 - val_recall: 0.6103
    Epoch 25/100
    142/142 [==============================] - 42s 293ms/step - loss: 0.6681 - accuracy: 0.7170 - auc: 0.8777 - cohen_kappa: 0.5500 - f1_score: 0.7070 - precision: 0.7565 - recall: 0.6436 - val_loss: 0.6289 - val_accuracy: 0.7237 - val_auc: 0.8959 - val_cohen_kappa: 0.5695 - val_f1_score: 0.7044 - val_precision: 0.7529 - val_recall: 0.6660
    Epoch 26/100
    142/142 [==============================] - 42s 296ms/step - loss: 0.6161 - accuracy: 0.7601 - auc: 0.8973 - cohen_kappa: 0.6149 - f1_score: 0.7345 - precision: 0.7860 - recall: 0.7053 - val_loss: 0.5923 - val_accuracy: 0.7216 - val_auc: 0.9012 - val_cohen_kappa: 0.5598 - val_f1_score: 0.6997 - val_precision: 0.7610 - val_recall: 0.6763
    Epoch 27/100
    142/142 [==============================] - 41s 291ms/step - loss: 0.6212 - accuracy: 0.7169 - auc: 0.8906 - cohen_kappa: 0.5530 - f1_score: 0.6995 - precision: 0.7482 - recall: 0.6688 - val_loss: 0.6570 - val_accuracy: 0.6887 - val_auc: 0.8829 - val_cohen_kappa: 0.5039 - val_f1_score: 0.6587 - val_precision: 0.7245 - val_recall: 0.6289
    Epoch 28/100
    142/142 [==============================] - 42s 298ms/step - loss: 0.5774 - accuracy: 0.7524 - auc: 0.9084 - cohen_kappa: 0.6044 - f1_score: 0.7314 - precision: 0.7900 - recall: 0.7182 - val_loss: 0.6074 - val_accuracy: 0.8000 - val_auc: 0.9183 - val_cohen_kappa: 0.6984 - val_f1_score: 0.7910 - val_precision: 0.8504 - val_recall: 0.7031
    Epoch 29/100
    142/142 [==============================] - 42s 296ms/step - loss: 0.5940 - accuracy: 0.7542 - auc: 0.9032 - cohen_kappa: 0.6116 - f1_score: 0.7377 - precision: 0.7816 - recall: 0.7017 - val_loss: 0.5365 - val_accuracy: 0.7567 - val_auc: 0.9223 - val_cohen_kappa: 0.6177 - val_f1_score: 0.7396 - val_precision: 0.7915 - val_recall: 0.7278
    Epoch 30/100
    142/142 [==============================] - 42s 296ms/step - loss: 0.5015 - accuracy: 0.7906 - auc: 0.9314 - cohen_kappa: 0.6683 - f1_score: 0.7749 - precision: 0.8250 - recall: 0.7642 - val_loss: 0.5455 - val_accuracy: 0.7299 - val_auc: 0.9197 - val_cohen_kappa: 0.5760 - val_f1_score: 0.7044 - val_precision: 0.7897 - val_recall: 0.6969
    Epoch 31/100
    142/142 [==============================] - 41s 290ms/step - loss: 0.5152 - accuracy: 0.7949 - auc: 0.9286 - cohen_kappa: 0.6727 - f1_score: 0.7773 - precision: 0.8202 - recall: 0.7656 - val_loss: 0.5805 - val_accuracy: 0.7773 - val_auc: 0.9107 - val_cohen_kappa: 0.6663 - val_f1_score: 0.7740 - val_precision: 0.8169 - val_recall: 0.6990
    Epoch 32/100
    142/142 [==============================] - 41s 292ms/step - loss: 0.5032 - accuracy: 0.7881 - auc: 0.9304 - cohen_kappa: 0.6659 - f1_score: 0.7756 - precision: 0.8113 - recall: 0.7666 - val_loss: 0.4439 - val_accuracy: 0.8351 - val_auc: 0.9529 - val_cohen_kappa: 0.7460 - val_f1_score: 0.8280 - val_precision: 0.8625 - val_recall: 0.8021
    Epoch 33/100
    142/142 [==============================] - 42s 294ms/step - loss: 0.5419 - accuracy: 0.7682 - auc: 0.9184 - cohen_kappa: 0.6317 - f1_score: 0.7474 - precision: 0.7937 - recall: 0.7465 - val_loss: 0.5534 - val_accuracy: 0.7649 - val_auc: 0.9202 - val_cohen_kappa: 0.6262 - val_f1_score: 0.7275 - val_precision: 0.7848 - val_recall: 0.7443
    Epoch 34/100
    142/142 [==============================] - 44s 308ms/step - loss: 0.4761 - accuracy: 0.8146 - auc: 0.9377 - cohen_kappa: 0.7040 - f1_score: 0.7910 - precision: 0.8313 - recall: 0.7810 - val_loss: 0.4235 - val_accuracy: 0.8247 - val_auc: 0.9544 - val_cohen_kappa: 0.7272 - val_f1_score: 0.8192 - val_precision: 0.8584 - val_recall: 0.8000
    Epoch 35/100
    142/142 [==============================] - 42s 292ms/step - loss: 0.5057 - accuracy: 0.7830 - auc: 0.9290 - cohen_kappa: 0.6615 - f1_score: 0.7722 - precision: 0.8038 - recall: 0.7581 - val_loss: 0.4424 - val_accuracy: 0.8124 - val_auc: 0.9454 - val_cohen_kappa: 0.7085 - val_f1_score: 0.8074 - val_precision: 0.8280 - val_recall: 0.7938
    Epoch 36/100
    142/142 [==============================] - 42s 294ms/step - loss: 0.4434 - accuracy: 0.8183 - auc: 0.9457 - cohen_kappa: 0.7148 - f1_score: 0.8041 - precision: 0.8437 - recall: 0.7952 - val_loss: 0.3634 - val_accuracy: 0.8990 - val_auc: 0.9703 - val_cohen_kappa: 0.8450 - val_f1_score: 0.8986 - val_precision: 0.9212 - val_recall: 0.8680
    Epoch 37/100
    142/142 [==============================] - 42s 295ms/step - loss: 0.3809 - accuracy: 0.8507 - auc: 0.9593 - cohen_kappa: 0.7640 - f1_score: 0.8398 - precision: 0.8628 - recall: 0.8339 - val_loss: 0.3393 - val_accuracy: 0.8784 - val_auc: 0.9701 - val_cohen_kappa: 0.8139 - val_f1_score: 0.8755 - val_precision: 0.8870 - val_recall: 0.8577
    Epoch 38/100
    142/142 [==============================] - 42s 293ms/step - loss: 0.3342 - accuracy: 0.8813 - auc: 0.9691 - cohen_kappa: 0.8134 - f1_score: 0.8745 - precision: 0.8871 - recall: 0.8682 - val_loss: 0.3002 - val_accuracy: 0.9216 - val_auc: 0.9770 - val_cohen_kappa: 0.8795 - val_f1_score: 0.9211 - val_precision: 0.9239 - val_recall: 0.9010
    Epoch 39/100
    142/142 [==============================] - 42s 292ms/step - loss: 0.3603 - accuracy: 0.8748 - auc: 0.9631 - cohen_kappa: 0.8040 - f1_score: 0.8649 - precision: 0.8836 - recall: 0.8585 - val_loss: 0.3309 - val_accuracy: 0.8722 - val_auc: 0.9708 - val_cohen_kappa: 0.8004 - val_f1_score: 0.8552 - val_precision: 0.8776 - val_recall: 0.8577
    Epoch 40/100
    142/142 [==============================] - 44s 309ms/step - loss: 0.3229 - accuracy: 0.8692 - auc: 0.9708 - cohen_kappa: 0.7946 - f1_score: 0.8619 - precision: 0.8806 - recall: 0.8557 - val_loss: 0.3484 - val_accuracy: 0.8515 - val_auc: 0.9679 - val_cohen_kappa: 0.7684 - val_f1_score: 0.8353 - val_precision: 0.8583 - val_recall: 0.8495
    Epoch 41/100
    142/142 [==============================] - 42s 298ms/step - loss: 0.3356 - accuracy: 0.8649 - auc: 0.9683 - cohen_kappa: 0.7879 - f1_score: 0.8574 - precision: 0.8729 - recall: 0.8498 - val_loss: 0.3083 - val_accuracy: 0.8784 - val_auc: 0.9731 - val_cohen_kappa: 0.8138 - val_f1_score: 0.8744 - val_precision: 0.8953 - val_recall: 0.8639
    Epoch 42/100
    142/142 [==============================] - 42s 294ms/step - loss: 0.3319 - accuracy: 0.8633 - auc: 0.9688 - cohen_kappa: 0.7873 - f1_score: 0.8554 - precision: 0.8747 - recall: 0.8527 - val_loss: 0.3424 - val_accuracy: 0.8680 - val_auc: 0.9708 - val_cohen_kappa: 0.7942 - val_f1_score: 0.8619 - val_precision: 0.8700 - val_recall: 0.8557
    Epoch 43/100
    142/142 [==============================] - 42s 295ms/step - loss: 0.2826 - accuracy: 0.8914 - auc: 0.9776 - cohen_kappa: 0.8281 - f1_score: 0.8839 - precision: 0.8969 - recall: 0.8823 - val_loss: 0.2919 - val_accuracy: 0.8928 - val_auc: 0.9777 - val_cohen_kappa: 0.8334 - val_f1_score: 0.8928 - val_precision: 0.8944 - val_recall: 0.8907
    Epoch 44/100
    142/142 [==============================] - 42s 297ms/step - loss: 0.3491 - accuracy: 0.8685 - auc: 0.9666 - cohen_kappa: 0.7957 - f1_score: 0.8615 - precision: 0.8822 - recall: 0.8542 - val_loss: 0.1878 - val_accuracy: 0.9423 - val_auc: 0.9890 - val_cohen_kappa: 0.9115 - val_f1_score: 0.9414 - val_precision: 0.9479 - val_recall: 0.9381
    Epoch 45/100
    142/142 [==============================] - 42s 296ms/step - loss: 0.2181 - accuracy: 0.9213 - auc: 0.9861 - cohen_kappa: 0.8771 - f1_score: 0.9179 - precision: 0.9242 - recall: 0.9142 - val_loss: 0.3003 - val_accuracy: 0.8948 - val_auc: 0.9754 - val_cohen_kappa: 0.8363 - val_f1_score: 0.8833 - val_precision: 0.9000 - val_recall: 0.8907
    Epoch 46/100
    142/142 [==============================] - 42s 296ms/step - loss: 0.2464 - accuracy: 0.9037 - auc: 0.9822 - cohen_kappa: 0.8493 - f1_score: 0.8955 - precision: 0.9129 - recall: 0.8981 - val_loss: 0.1470 - val_accuracy: 0.9588 - val_auc: 0.9919 - val_cohen_kappa: 0.9366 - val_f1_score: 0.9590 - val_precision: 0.9587 - val_recall: 0.9567
    Epoch 47/100
    142/142 [==============================] - 44s 310ms/step - loss: 0.2393 - accuracy: 0.9094 - auc: 0.9836 - cohen_kappa: 0.8579 - f1_score: 0.9018 - precision: 0.9140 - recall: 0.9045 - val_loss: 0.2030 - val_accuracy: 0.9237 - val_auc: 0.9880 - val_cohen_kappa: 0.8824 - val_f1_score: 0.9215 - val_precision: 0.9256 - val_recall: 0.9237
    Epoch 48/100
    142/142 [==============================] - 42s 298ms/step - loss: 0.1958 - accuracy: 0.9271 - auc: 0.9887 - cohen_kappa: 0.8861 - f1_score: 0.9225 - precision: 0.9310 - recall: 0.9209 - val_loss: 0.1745 - val_accuracy: 0.9237 - val_auc: 0.9902 - val_cohen_kappa: 0.8820 - val_f1_score: 0.9250 - val_precision: 0.9236 - val_recall: 0.9216
    Epoch 49/100
    142/142 [==============================] - 42s 296ms/step - loss: 0.2077 - accuracy: 0.9132 - auc: 0.9874 - cohen_kappa: 0.8630 - f1_score: 0.9052 - precision: 0.9148 - recall: 0.9074 - val_loss: 0.1440 - val_accuracy: 0.9505 - val_auc: 0.9945 - val_cohen_kappa: 0.9244 - val_f1_score: 0.9524 - val_precision: 0.9545 - val_recall: 0.9505
    Epoch 50/100
    142/142 [==============================] - 42s 294ms/step - loss: 0.1666 - accuracy: 0.9335 - auc: 0.9917 - cohen_kappa: 0.8973 - f1_score: 0.9323 - precision: 0.9407 - recall: 0.9318 - val_loss: 0.3575 - val_accuracy: 0.8845 - val_auc: 0.9723 - val_cohen_kappa: 0.8195 - val_f1_score: 0.8635 - val_precision: 0.8880 - val_recall: 0.8825
    Epoch 51/100
    142/142 [==============================] - 42s 296ms/step - loss: 0.1840 - accuracy: 0.9310 - auc: 0.9893 - cohen_kappa: 0.8921 - f1_score: 0.9244 - precision: 0.9348 - recall: 0.9280 - val_loss: 0.1130 - val_accuracy: 0.9691 - val_auc: 0.9941 - val_cohen_kappa: 0.9526 - val_f1_score: 0.9685 - val_precision: 0.9691 - val_recall: 0.9691
    Epoch 52/100
    142/142 [==============================] - 42s 298ms/step - loss: 0.1324 - accuracy: 0.9479 - auc: 0.9947 - cohen_kappa: 0.9183 - f1_score: 0.9419 - precision: 0.9503 - recall: 0.9469 - val_loss: 0.1037 - val_accuracy: 0.9629 - val_auc: 0.9956 - val_cohen_kappa: 0.9430 - val_f1_score: 0.9648 - val_precision: 0.9628 - val_recall: 0.9608
    Epoch 53/100
    142/142 [==============================] - 42s 296ms/step - loss: 0.1495 - accuracy: 0.9501 - auc: 0.9923 - cohen_kappa: 0.9220 - f1_score: 0.9471 - precision: 0.9531 - recall: 0.9478 - val_loss: 0.0747 - val_accuracy: 0.9711 - val_auc: 0.9981 - val_cohen_kappa: 0.9557 - val_f1_score: 0.9703 - val_precision: 0.9711 - val_recall: 0.9711
    Epoch 54/100
    142/142 [==============================] - 45s 313ms/step - loss: 0.0820 - accuracy: 0.9695 - auc: 0.9979 - cohen_kappa: 0.9526 - f1_score: 0.9675 - precision: 0.9695 - recall: 0.9683 - val_loss: 0.0882 - val_accuracy: 0.9670 - val_auc: 0.9962 - val_cohen_kappa: 0.9492 - val_f1_score: 0.9678 - val_precision: 0.9669 - val_recall: 0.9649
    Epoch 55/100
    142/142 [==============================] - 40s 282ms/step - loss: 0.1170 - accuracy: 0.9554 - auc: 0.9956 - cohen_kappa: 0.9294 - f1_score: 0.9534 - precision: 0.9566 - recall: 0.9538 - val_loss: 0.0906 - val_accuracy: 0.9794 - val_auc: 0.9937 - val_cohen_kappa: 0.9684 - val_f1_score: 0.9788 - val_precision: 0.9794 - val_recall: 0.9794
    Epoch 56/100
    142/142 [==============================] - 42s 295ms/step - loss: 0.0814 - accuracy: 0.9644 - auc: 0.9980 - cohen_kappa: 0.9449 - f1_score: 0.9618 - precision: 0.9670 - recall: 0.9644 - val_loss: 0.0725 - val_accuracy: 0.9794 - val_auc: 0.9980 - val_cohen_kappa: 0.9683 - val_f1_score: 0.9781 - val_precision: 0.9794 - val_recall: 0.9794
    Epoch 57/100
    142/142 [==============================] - 40s 278ms/step - loss: 0.1352 - accuracy: 0.9509 - auc: 0.9938 - cohen_kappa: 0.9235 - f1_score: 0.9478 - precision: 0.9523 - recall: 0.9479 - val_loss: 0.3284 - val_accuracy: 0.8887 - val_auc: 0.9776 - val_cohen_kappa: 0.8260 - val_f1_score: 0.8824 - val_precision: 0.8903 - val_recall: 0.8866
    Epoch 58/100
    142/142 [==============================] - 42s 297ms/step - loss: 0.0885 - accuracy: 0.9659 - auc: 0.9974 - cohen_kappa: 0.9467 - f1_score: 0.9638 - precision: 0.9681 - recall: 0.9648 - val_loss: 0.0963 - val_accuracy: 0.9732 - val_auc: 0.9947 - val_cohen_kappa: 0.9588 - val_f1_score: 0.9739 - val_precision: 0.9732 - val_recall: 0.9732
    Epoch 59/100
    142/142 [==============================] - 43s 301ms/step - loss: 0.0766 - accuracy: 0.9710 - auc: 0.9983 - cohen_kappa: 0.9545 - f1_score: 0.9704 - precision: 0.9712 - recall: 0.9706 - val_loss: 0.3159 - val_accuracy: 0.9052 - val_auc: 0.9781 - val_cohen_kappa: 0.8527 - val_f1_score: 0.9066 - val_precision: 0.9052 - val_recall: 0.9052
    Epoch 60/100
    142/142 [==============================] - 39s 277ms/step - loss: 0.1266 - accuracy: 0.9554 - auc: 0.9940 - cohen_kappa: 0.9302 - f1_score: 0.9491 - precision: 0.9562 - recall: 0.9545 - val_loss: 0.0695 - val_accuracy: 0.9814 - val_auc: 0.9972 - val_cohen_kappa: 0.9715 - val_f1_score: 0.9811 - val_precision: 0.9835 - val_recall: 0.9814
    Epoch 61/100
    142/142 [==============================] - 42s 299ms/step - loss: 0.0746 - accuracy: 0.9734 - auc: 0.9982 - cohen_kappa: 0.9578 - f1_score: 0.9738 - precision: 0.9746 - recall: 0.9734 - val_loss: 0.1187 - val_accuracy: 0.9649 - val_auc: 0.9933 - val_cohen_kappa: 0.9461 - val_f1_score: 0.9660 - val_precision: 0.9649 - val_recall: 0.9649
    Epoch 62/100
    142/142 [==============================] - 40s 278ms/step - loss: 0.0970 - accuracy: 0.9632 - auc: 0.9970 - cohen_kappa: 0.9424 - f1_score: 0.9614 - precision: 0.9634 - recall: 0.9623 - val_loss: 0.0961 - val_accuracy: 0.9711 - val_auc: 0.9965 - val_cohen_kappa: 0.9555 - val_f1_score: 0.9689 - val_precision: 0.9772 - val_recall: 0.9711
    Epoch 63/100
    142/142 [==============================] - 43s 302ms/step - loss: 0.1109 - accuracy: 0.9574 - auc: 0.9962 - cohen_kappa: 0.9336 - f1_score: 0.9542 - precision: 0.9610 - recall: 0.9574 - val_loss: 0.0457 - val_accuracy: 0.9876 - val_auc: 0.9980 - val_cohen_kappa: 0.9810 - val_f1_score: 0.9876 - val_precision: 0.9876 - val_recall: 0.9876
    Epoch 64/100
    142/142 [==============================] - 42s 297ms/step - loss: 0.0584 - accuracy: 0.9803 - auc: 0.9986 - cohen_kappa: 0.9692 - f1_score: 0.9786 - precision: 0.9814 - recall: 0.9803 - val_loss: 0.1149 - val_accuracy: 0.9546 - val_auc: 0.9956 - val_cohen_kappa: 0.9303 - val_f1_score: 0.9552 - val_precision: 0.9586 - val_recall: 0.9546
    Epoch 65/100
    142/142 [==============================] - 39s 276ms/step - loss: 0.0654 - accuracy: 0.9761 - auc: 0.9987 - cohen_kappa: 0.9628 - f1_score: 0.9743 - precision: 0.9767 - recall: 0.9752 - val_loss: 0.0836 - val_accuracy: 0.9794 - val_auc: 0.9970 - val_cohen_kappa: 0.9683 - val_f1_score: 0.9795 - val_precision: 0.9794 - val_recall: 0.9794
    Epoch 66/100
    142/142 [==============================] - 43s 300ms/step - loss: 0.0666 - accuracy: 0.9715 - auc: 0.9986 - cohen_kappa: 0.9556 - f1_score: 0.9696 - precision: 0.9728 - recall: 0.9713 - val_loss: 0.1330 - val_accuracy: 0.9608 - val_auc: 0.9936 - val_cohen_kappa: 0.9398 - val_f1_score: 0.9615 - val_precision: 0.9628 - val_recall: 0.9608
    Epoch 67/100
    142/142 [==============================] - 40s 282ms/step - loss: 0.0622 - accuracy: 0.9740 - auc: 0.9989 - cohen_kappa: 0.9594 - f1_score: 0.9721 - precision: 0.9751 - recall: 0.9738 - val_loss: 0.0397 - val_accuracy: 0.9897 - val_auc: 0.9981 - val_cohen_kappa: 0.9842 - val_f1_score: 0.9900 - val_precision: 0.9897 - val_recall: 0.9897
    Epoch 68/100
    142/142 [==============================] - 42s 298ms/step - loss: 0.0708 - accuracy: 0.9773 - auc: 0.9969 - cohen_kappa: 0.9642 - f1_score: 0.9763 - precision: 0.9774 - recall: 0.9770 - val_loss: 0.0480 - val_accuracy: 0.9918 - val_auc: 0.9979 - val_cohen_kappa: 0.9873 - val_f1_score: 0.9912 - val_precision: 0.9918 - val_recall: 0.9918
    Epoch 69/100
    142/142 [==============================] - 42s 296ms/step - loss: 0.0628 - accuracy: 0.9773 - auc: 0.9983 - cohen_kappa: 0.9645 - f1_score: 0.9766 - precision: 0.9785 - recall: 0.9770 - val_loss: 0.0602 - val_accuracy: 0.9876 - val_auc: 0.9964 - val_cohen_kappa: 0.9810 - val_f1_score: 0.9866 - val_precision: 0.9876 - val_recall: 0.9856
    Epoch 70/100
    142/142 [==============================] - 40s 279ms/step - loss: 0.0538 - accuracy: 0.9849 - auc: 0.9983 - cohen_kappa: 0.9764 - f1_score: 0.9845 - precision: 0.9854 - recall: 0.9849 - val_loss: 0.0424 - val_accuracy: 0.9876 - val_auc: 0.9980 - val_cohen_kappa: 0.9810 - val_f1_score: 0.9876 - val_precision: 0.9876 - val_recall: 0.9876
    Epoch 71/100
    142/142 [==============================] - 43s 300ms/step - loss: 0.0675 - accuracy: 0.9735 - auc: 0.9982 - cohen_kappa: 0.9588 - f1_score: 0.9730 - precision: 0.9735 - recall: 0.9728 - val_loss: 0.0595 - val_accuracy: 0.9794 - val_auc: 0.9977 - val_cohen_kappa: 0.9683 - val_f1_score: 0.9776 - val_precision: 0.9794 - val_recall: 0.9794
    Epoch 72/100
    142/142 [==============================] - 40s 279ms/step - loss: 0.0850 - accuracy: 0.9695 - auc: 0.9973 - cohen_kappa: 0.9525 - f1_score: 0.9673 - precision: 0.9704 - recall: 0.9690 - val_loss: 0.1250 - val_accuracy: 0.9608 - val_auc: 0.9940 - val_cohen_kappa: 0.9396 - val_f1_score: 0.9592 - val_precision: 0.9607 - val_recall: 0.9588
    Epoch 73/100
    142/142 [==============================] - 42s 295ms/step - loss: 0.0453 - accuracy: 0.9832 - auc: 0.9992 - cohen_kappa: 0.9736 - f1_score: 0.9826 - precision: 0.9841 - recall: 0.9809 - val_loss: 0.0490 - val_accuracy: 0.9876 - val_auc: 0.9979 - val_cohen_kappa: 0.9810 - val_f1_score: 0.9870 - val_precision: 0.9876 - val_recall: 0.9876
    Epoch 74/100
    142/142 [==============================] - 43s 303ms/step - loss: 0.0279 - accuracy: 0.9928 - auc: 0.9997 - cohen_kappa: 0.9886 - f1_score: 0.9925 - precision: 0.9932 - recall: 0.9924 - val_loss: 0.0589 - val_accuracy: 0.9897 - val_auc: 0.9963 - val_cohen_kappa: 0.9842 - val_f1_score: 0.9894 - val_precision: 0.9897 - val_recall: 0.9897
    Epoch 75/100
    142/142 [==============================] - 40s 279ms/step - loss: 0.0241 - accuracy: 0.9946 - auc: 0.9998 - cohen_kappa: 0.9916 - f1_score: 0.9950 - precision: 0.9946 - recall: 0.9945 - val_loss: 0.0547 - val_accuracy: 0.9876 - val_auc: 0.9977 - val_cohen_kappa: 0.9810 - val_f1_score: 0.9870 - val_precision: 0.9876 - val_recall: 0.9876
    Epoch 76/100
    142/142 [==============================] - 43s 301ms/step - loss: 0.0183 - accuracy: 0.9959 - auc: 1.0000 - cohen_kappa: 0.9936 - f1_score: 0.9957 - precision: 0.9959 - recall: 0.9955 - val_loss: 0.0458 - val_accuracy: 0.9876 - val_auc: 0.9980 - val_cohen_kappa: 0.9810 - val_f1_score: 0.9870 - val_precision: 0.9876 - val_recall: 0.9876
    Epoch 77/100
    142/142 [==============================] - 39s 277ms/step - loss: 0.0303 - accuracy: 0.9904 - auc: 0.9993 - cohen_kappa: 0.9850 - f1_score: 0.9894 - precision: 0.9904 - recall: 0.9904 - val_loss: 0.0461 - val_accuracy: 0.9897 - val_auc: 0.9979 - val_cohen_kappa: 0.9842 - val_f1_score: 0.9888 - val_precision: 0.9897 - val_recall: 0.9897
    Epoch 78/100
    142/142 [==============================] - 43s 303ms/step - loss: 0.0231 - accuracy: 0.9955 - auc: 0.9984 - cohen_kappa: 0.9931 - f1_score: 0.9951 - precision: 0.9955 - recall: 0.9955 - val_loss: 0.0472 - val_accuracy: 0.9897 - val_auc: 0.9979 - val_cohen_kappa: 0.9842 - val_f1_score: 0.9888 - val_precision: 0.9897 - val_recall: 0.9897
    Epoch 79/100
    142/142 [==============================] - 43s 300ms/step - loss: 0.0262 - accuracy: 0.9934 - auc: 0.9991 - cohen_kappa: 0.9897 - f1_score: 0.9932 - precision: 0.9934 - recall: 0.9934 - val_loss: 0.0466 - val_accuracy: 0.9918 - val_auc: 0.9979 - val_cohen_kappa: 0.9873 - val_f1_score: 0.9912 - val_precision: 0.9918 - val_recall: 0.9918
    Epoch 80/100
    142/142 [==============================] - 40s 279ms/step - loss: 0.0173 - accuracy: 0.9950 - auc: 0.9996 - cohen_kappa: 0.9922 - f1_score: 0.9944 - precision: 0.9950 - recall: 0.9950 - val_loss: 0.0457 - val_accuracy: 0.9897 - val_auc: 0.9980 - val_cohen_kappa: 0.9842 - val_f1_score: 0.9894 - val_precision: 0.9897 - val_recall: 0.9897
    Epoch 81/100
    142/142 [==============================] - 42s 299ms/step - loss: 0.0204 - accuracy: 0.9944 - auc: 0.9999 - cohen_kappa: 0.9913 - f1_score: 0.9941 - precision: 0.9944 - recall: 0.9944 - val_loss: 0.0452 - val_accuracy: 0.9897 - val_auc: 0.9980 - val_cohen_kappa: 0.9842 - val_f1_score: 0.9894 - val_precision: 0.9897 - val_recall: 0.9897
    Epoch 82/100
    142/142 [==============================] - 40s 280ms/step - loss: 0.0187 - accuracy: 0.9918 - auc: 0.9999 - cohen_kappa: 0.9871 - f1_score: 0.9915 - precision: 0.9918 - recall: 0.9918 - val_loss: 0.0443 - val_accuracy: 0.9897 - val_auc: 0.9980 - val_cohen_kappa: 0.9842 - val_f1_score: 0.9894 - val_precision: 0.9897 - val_recall: 0.9897
    Epoch 83/100
    142/142 [==============================] - 43s 298ms/step - loss: 0.0130 - accuracy: 0.9973 - auc: 1.0000 - cohen_kappa: 0.9958 - f1_score: 0.9970 - precision: 0.9973 - recall: 0.9970 - val_loss: 0.0443 - val_accuracy: 0.9897 - val_auc: 0.9980 - val_cohen_kappa: 0.9842 - val_f1_score: 0.9894 - val_precision: 0.9897 - val_recall: 0.9897
    Epoch 00083: early stopping
    


```python
evaluate_model(model, history, test_generator)
```

    
    Test set accuracy: 0.985567033290863 
    
    31/31 [==============================] - 3s 79ms/step
    
                  precision    recall  f1-score   support
    
             AMD       0.98      0.98      0.98       121
             DME       0.99      0.97      0.98       155
          NORMAL       0.99      1.00      0.99       209
    
        accuracy                           0.99       485
       macro avg       0.98      0.98      0.98       485
    weighted avg       0.99      0.99      0.99       485
    
    


![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_31_1.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_31_2.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_31_3.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_31_4.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_31_5.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_31_6.png)


    ROC AUC score: 0.9995329217497156
    


```python
inception_model = generate_model('inceptionv3', 3)

```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
    87916544/87910968 [==============================] - 1s 0us/step
    


```python
inception_model, inception_history = train_model(inception_model, train_generator, val_generator, 20, tf.keras.optimizers.SGD(lr=0.001, momentum=0.9), metrics)
```

    Model: "model_4"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_4 (InputLayer)            [(None, 224, 224, 3) 0                                            
    __________________________________________________________________________________________________
    conv2d_212 (Conv2D)             (None, 111, 111, 32) 864         input_4[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization_203 (BatchN (None, 111, 111, 32) 96          conv2d_212[0][0]                 
    __________________________________________________________________________________________________
    activation_203 (Activation)     (None, 111, 111, 32) 0           batch_normalization_203[0][0]    
    __________________________________________________________________________________________________
    conv2d_213 (Conv2D)             (None, 109, 109, 32) 9216        activation_203[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_204 (BatchN (None, 109, 109, 32) 96          conv2d_213[0][0]                 
    __________________________________________________________________________________________________
    activation_204 (Activation)     (None, 109, 109, 32) 0           batch_normalization_204[0][0]    
    __________________________________________________________________________________________________
    conv2d_214 (Conv2D)             (None, 109, 109, 64) 18432       activation_204[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_205 (BatchN (None, 109, 109, 64) 192         conv2d_214[0][0]                 
    __________________________________________________________________________________________________
    activation_205 (Activation)     (None, 109, 109, 64) 0           batch_normalization_205[0][0]    
    __________________________________________________________________________________________________
    max_pooling2d_7 (MaxPooling2D)  (None, 54, 54, 64)   0           activation_205[0][0]             
    __________________________________________________________________________________________________
    conv2d_215 (Conv2D)             (None, 54, 54, 80)   5120        max_pooling2d_7[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_206 (BatchN (None, 54, 54, 80)   240         conv2d_215[0][0]                 
    __________________________________________________________________________________________________
    activation_206 (Activation)     (None, 54, 54, 80)   0           batch_normalization_206[0][0]    
    __________________________________________________________________________________________________
    conv2d_216 (Conv2D)             (None, 52, 52, 192)  138240      activation_206[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_207 (BatchN (None, 52, 52, 192)  576         conv2d_216[0][0]                 
    __________________________________________________________________________________________________
    activation_207 (Activation)     (None, 52, 52, 192)  0           batch_normalization_207[0][0]    
    __________________________________________________________________________________________________
    max_pooling2d_8 (MaxPooling2D)  (None, 25, 25, 192)  0           activation_207[0][0]             
    __________________________________________________________________________________________________
    conv2d_220 (Conv2D)             (None, 25, 25, 64)   12288       max_pooling2d_8[0][0]            
    __________________________________________________________________________________________________
    batch_normalization_211 (BatchN (None, 25, 25, 64)   192         conv2d_220[0][0]                 
    __________________________________________________________________________________________________
    activation_211 (Activation)     (None, 25, 25, 64)   0           batch_normalization_211[0][0]    
    __________________________________________________________________________________________________
    conv2d_218 (Conv2D)             (None, 25, 25, 48)   9216        max_pooling2d_8[0][0]            
    __________________________________________________________________________________________________
    conv2d_221 (Conv2D)             (None, 25, 25, 96)   55296       activation_211[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_209 (BatchN (None, 25, 25, 48)   144         conv2d_218[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_212 (BatchN (None, 25, 25, 96)   288         conv2d_221[0][0]                 
    __________________________________________________________________________________________________
    activation_209 (Activation)     (None, 25, 25, 48)   0           batch_normalization_209[0][0]    
    __________________________________________________________________________________________________
    activation_212 (Activation)     (None, 25, 25, 96)   0           batch_normalization_212[0][0]    
    __________________________________________________________________________________________________
    average_pooling2d_1 (AveragePoo (None, 25, 25, 192)  0           max_pooling2d_8[0][0]            
    __________________________________________________________________________________________________
    conv2d_217 (Conv2D)             (None, 25, 25, 64)   12288       max_pooling2d_8[0][0]            
    __________________________________________________________________________________________________
    conv2d_219 (Conv2D)             (None, 25, 25, 64)   76800       activation_209[0][0]             
    __________________________________________________________________________________________________
    conv2d_222 (Conv2D)             (None, 25, 25, 96)   82944       activation_212[0][0]             
    __________________________________________________________________________________________________
    conv2d_223 (Conv2D)             (None, 25, 25, 32)   6144        average_pooling2d_1[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_208 (BatchN (None, 25, 25, 64)   192         conv2d_217[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_210 (BatchN (None, 25, 25, 64)   192         conv2d_219[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_213 (BatchN (None, 25, 25, 96)   288         conv2d_222[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_214 (BatchN (None, 25, 25, 32)   96          conv2d_223[0][0]                 
    __________________________________________________________________________________________________
    activation_208 (Activation)     (None, 25, 25, 64)   0           batch_normalization_208[0][0]    
    __________________________________________________________________________________________________
    activation_210 (Activation)     (None, 25, 25, 64)   0           batch_normalization_210[0][0]    
    __________________________________________________________________________________________________
    activation_213 (Activation)     (None, 25, 25, 96)   0           batch_normalization_213[0][0]    
    __________________________________________________________________________________________________
    activation_214 (Activation)     (None, 25, 25, 32)   0           batch_normalization_214[0][0]    
    __________________________________________________________________________________________________
    mixed0 (Concatenate)            (None, 25, 25, 256)  0           activation_208[0][0]             
                                                                     activation_210[0][0]             
                                                                     activation_213[0][0]             
                                                                     activation_214[0][0]             
    __________________________________________________________________________________________________
    conv2d_227 (Conv2D)             (None, 25, 25, 64)   16384       mixed0[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_218 (BatchN (None, 25, 25, 64)   192         conv2d_227[0][0]                 
    __________________________________________________________________________________________________
    activation_218 (Activation)     (None, 25, 25, 64)   0           batch_normalization_218[0][0]    
    __________________________________________________________________________________________________
    conv2d_225 (Conv2D)             (None, 25, 25, 48)   12288       mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_228 (Conv2D)             (None, 25, 25, 96)   55296       activation_218[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_216 (BatchN (None, 25, 25, 48)   144         conv2d_225[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_219 (BatchN (None, 25, 25, 96)   288         conv2d_228[0][0]                 
    __________________________________________________________________________________________________
    activation_216 (Activation)     (None, 25, 25, 48)   0           batch_normalization_216[0][0]    
    __________________________________________________________________________________________________
    activation_219 (Activation)     (None, 25, 25, 96)   0           batch_normalization_219[0][0]    
    __________________________________________________________________________________________________
    average_pooling2d_2 (AveragePoo (None, 25, 25, 256)  0           mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_224 (Conv2D)             (None, 25, 25, 64)   16384       mixed0[0][0]                     
    __________________________________________________________________________________________________
    conv2d_226 (Conv2D)             (None, 25, 25, 64)   76800       activation_216[0][0]             
    __________________________________________________________________________________________________
    conv2d_229 (Conv2D)             (None, 25, 25, 96)   82944       activation_219[0][0]             
    __________________________________________________________________________________________________
    conv2d_230 (Conv2D)             (None, 25, 25, 64)   16384       average_pooling2d_2[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_215 (BatchN (None, 25, 25, 64)   192         conv2d_224[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_217 (BatchN (None, 25, 25, 64)   192         conv2d_226[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_220 (BatchN (None, 25, 25, 96)   288         conv2d_229[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_221 (BatchN (None, 25, 25, 64)   192         conv2d_230[0][0]                 
    __________________________________________________________________________________________________
    activation_215 (Activation)     (None, 25, 25, 64)   0           batch_normalization_215[0][0]    
    __________________________________________________________________________________________________
    activation_217 (Activation)     (None, 25, 25, 64)   0           batch_normalization_217[0][0]    
    __________________________________________________________________________________________________
    activation_220 (Activation)     (None, 25, 25, 96)   0           batch_normalization_220[0][0]    
    __________________________________________________________________________________________________
    activation_221 (Activation)     (None, 25, 25, 64)   0           batch_normalization_221[0][0]    
    __________________________________________________________________________________________________
    mixed1 (Concatenate)            (None, 25, 25, 288)  0           activation_215[0][0]             
                                                                     activation_217[0][0]             
                                                                     activation_220[0][0]             
                                                                     activation_221[0][0]             
    __________________________________________________________________________________________________
    conv2d_234 (Conv2D)             (None, 25, 25, 64)   18432       mixed1[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_225 (BatchN (None, 25, 25, 64)   192         conv2d_234[0][0]                 
    __________________________________________________________________________________________________
    activation_225 (Activation)     (None, 25, 25, 64)   0           batch_normalization_225[0][0]    
    __________________________________________________________________________________________________
    conv2d_232 (Conv2D)             (None, 25, 25, 48)   13824       mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_235 (Conv2D)             (None, 25, 25, 96)   55296       activation_225[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_223 (BatchN (None, 25, 25, 48)   144         conv2d_232[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_226 (BatchN (None, 25, 25, 96)   288         conv2d_235[0][0]                 
    __________________________________________________________________________________________________
    activation_223 (Activation)     (None, 25, 25, 48)   0           batch_normalization_223[0][0]    
    __________________________________________________________________________________________________
    activation_226 (Activation)     (None, 25, 25, 96)   0           batch_normalization_226[0][0]    
    __________________________________________________________________________________________________
    average_pooling2d_3 (AveragePoo (None, 25, 25, 288)  0           mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_231 (Conv2D)             (None, 25, 25, 64)   18432       mixed1[0][0]                     
    __________________________________________________________________________________________________
    conv2d_233 (Conv2D)             (None, 25, 25, 64)   76800       activation_223[0][0]             
    __________________________________________________________________________________________________
    conv2d_236 (Conv2D)             (None, 25, 25, 96)   82944       activation_226[0][0]             
    __________________________________________________________________________________________________
    conv2d_237 (Conv2D)             (None, 25, 25, 64)   18432       average_pooling2d_3[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_222 (BatchN (None, 25, 25, 64)   192         conv2d_231[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_224 (BatchN (None, 25, 25, 64)   192         conv2d_233[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_227 (BatchN (None, 25, 25, 96)   288         conv2d_236[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_228 (BatchN (None, 25, 25, 64)   192         conv2d_237[0][0]                 
    __________________________________________________________________________________________________
    activation_222 (Activation)     (None, 25, 25, 64)   0           batch_normalization_222[0][0]    
    __________________________________________________________________________________________________
    activation_224 (Activation)     (None, 25, 25, 64)   0           batch_normalization_224[0][0]    
    __________________________________________________________________________________________________
    activation_227 (Activation)     (None, 25, 25, 96)   0           batch_normalization_227[0][0]    
    __________________________________________________________________________________________________
    activation_228 (Activation)     (None, 25, 25, 64)   0           batch_normalization_228[0][0]    
    __________________________________________________________________________________________________
    mixed2 (Concatenate)            (None, 25, 25, 288)  0           activation_222[0][0]             
                                                                     activation_224[0][0]             
                                                                     activation_227[0][0]             
                                                                     activation_228[0][0]             
    __________________________________________________________________________________________________
    conv2d_239 (Conv2D)             (None, 25, 25, 64)   18432       mixed2[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_230 (BatchN (None, 25, 25, 64)   192         conv2d_239[0][0]                 
    __________________________________________________________________________________________________
    activation_230 (Activation)     (None, 25, 25, 64)   0           batch_normalization_230[0][0]    
    __________________________________________________________________________________________________
    conv2d_240 (Conv2D)             (None, 25, 25, 96)   55296       activation_230[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_231 (BatchN (None, 25, 25, 96)   288         conv2d_240[0][0]                 
    __________________________________________________________________________________________________
    activation_231 (Activation)     (None, 25, 25, 96)   0           batch_normalization_231[0][0]    
    __________________________________________________________________________________________________
    conv2d_238 (Conv2D)             (None, 12, 12, 384)  995328      mixed2[0][0]                     
    __________________________________________________________________________________________________
    conv2d_241 (Conv2D)             (None, 12, 12, 96)   82944       activation_231[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_229 (BatchN (None, 12, 12, 384)  1152        conv2d_238[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_232 (BatchN (None, 12, 12, 96)   288         conv2d_241[0][0]                 
    __________________________________________________________________________________________________
    activation_229 (Activation)     (None, 12, 12, 384)  0           batch_normalization_229[0][0]    
    __________________________________________________________________________________________________
    activation_232 (Activation)     (None, 12, 12, 96)   0           batch_normalization_232[0][0]    
    __________________________________________________________________________________________________
    max_pooling2d_9 (MaxPooling2D)  (None, 12, 12, 288)  0           mixed2[0][0]                     
    __________________________________________________________________________________________________
    mixed3 (Concatenate)            (None, 12, 12, 768)  0           activation_229[0][0]             
                                                                     activation_232[0][0]             
                                                                     max_pooling2d_9[0][0]            
    __________________________________________________________________________________________________
    conv2d_246 (Conv2D)             (None, 12, 12, 128)  98304       mixed3[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_237 (BatchN (None, 12, 12, 128)  384         conv2d_246[0][0]                 
    __________________________________________________________________________________________________
    activation_237 (Activation)     (None, 12, 12, 128)  0           batch_normalization_237[0][0]    
    __________________________________________________________________________________________________
    conv2d_247 (Conv2D)             (None, 12, 12, 128)  114688      activation_237[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_238 (BatchN (None, 12, 12, 128)  384         conv2d_247[0][0]                 
    __________________________________________________________________________________________________
    activation_238 (Activation)     (None, 12, 12, 128)  0           batch_normalization_238[0][0]    
    __________________________________________________________________________________________________
    conv2d_243 (Conv2D)             (None, 12, 12, 128)  98304       mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_248 (Conv2D)             (None, 12, 12, 128)  114688      activation_238[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_234 (BatchN (None, 12, 12, 128)  384         conv2d_243[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_239 (BatchN (None, 12, 12, 128)  384         conv2d_248[0][0]                 
    __________________________________________________________________________________________________
    activation_234 (Activation)     (None, 12, 12, 128)  0           batch_normalization_234[0][0]    
    __________________________________________________________________________________________________
    activation_239 (Activation)     (None, 12, 12, 128)  0           batch_normalization_239[0][0]    
    __________________________________________________________________________________________________
    conv2d_244 (Conv2D)             (None, 12, 12, 128)  114688      activation_234[0][0]             
    __________________________________________________________________________________________________
    conv2d_249 (Conv2D)             (None, 12, 12, 128)  114688      activation_239[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_235 (BatchN (None, 12, 12, 128)  384         conv2d_244[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_240 (BatchN (None, 12, 12, 128)  384         conv2d_249[0][0]                 
    __________________________________________________________________________________________________
    activation_235 (Activation)     (None, 12, 12, 128)  0           batch_normalization_235[0][0]    
    __________________________________________________________________________________________________
    activation_240 (Activation)     (None, 12, 12, 128)  0           batch_normalization_240[0][0]    
    __________________________________________________________________________________________________
    average_pooling2d_4 (AveragePoo (None, 12, 12, 768)  0           mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_242 (Conv2D)             (None, 12, 12, 192)  147456      mixed3[0][0]                     
    __________________________________________________________________________________________________
    conv2d_245 (Conv2D)             (None, 12, 12, 192)  172032      activation_235[0][0]             
    __________________________________________________________________________________________________
    conv2d_250 (Conv2D)             (None, 12, 12, 192)  172032      activation_240[0][0]             
    __________________________________________________________________________________________________
    conv2d_251 (Conv2D)             (None, 12, 12, 192)  147456      average_pooling2d_4[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_233 (BatchN (None, 12, 12, 192)  576         conv2d_242[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_236 (BatchN (None, 12, 12, 192)  576         conv2d_245[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_241 (BatchN (None, 12, 12, 192)  576         conv2d_250[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_242 (BatchN (None, 12, 12, 192)  576         conv2d_251[0][0]                 
    __________________________________________________________________________________________________
    activation_233 (Activation)     (None, 12, 12, 192)  0           batch_normalization_233[0][0]    
    __________________________________________________________________________________________________
    activation_236 (Activation)     (None, 12, 12, 192)  0           batch_normalization_236[0][0]    
    __________________________________________________________________________________________________
    activation_241 (Activation)     (None, 12, 12, 192)  0           batch_normalization_241[0][0]    
    __________________________________________________________________________________________________
    activation_242 (Activation)     (None, 12, 12, 192)  0           batch_normalization_242[0][0]    
    __________________________________________________________________________________________________
    mixed4 (Concatenate)            (None, 12, 12, 768)  0           activation_233[0][0]             
                                                                     activation_236[0][0]             
                                                                     activation_241[0][0]             
                                                                     activation_242[0][0]             
    __________________________________________________________________________________________________
    conv2d_256 (Conv2D)             (None, 12, 12, 160)  122880      mixed4[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_247 (BatchN (None, 12, 12, 160)  480         conv2d_256[0][0]                 
    __________________________________________________________________________________________________
    activation_247 (Activation)     (None, 12, 12, 160)  0           batch_normalization_247[0][0]    
    __________________________________________________________________________________________________
    conv2d_257 (Conv2D)             (None, 12, 12, 160)  179200      activation_247[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_248 (BatchN (None, 12, 12, 160)  480         conv2d_257[0][0]                 
    __________________________________________________________________________________________________
    activation_248 (Activation)     (None, 12, 12, 160)  0           batch_normalization_248[0][0]    
    __________________________________________________________________________________________________
    conv2d_253 (Conv2D)             (None, 12, 12, 160)  122880      mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_258 (Conv2D)             (None, 12, 12, 160)  179200      activation_248[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_244 (BatchN (None, 12, 12, 160)  480         conv2d_253[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_249 (BatchN (None, 12, 12, 160)  480         conv2d_258[0][0]                 
    __________________________________________________________________________________________________
    activation_244 (Activation)     (None, 12, 12, 160)  0           batch_normalization_244[0][0]    
    __________________________________________________________________________________________________
    activation_249 (Activation)     (None, 12, 12, 160)  0           batch_normalization_249[0][0]    
    __________________________________________________________________________________________________
    conv2d_254 (Conv2D)             (None, 12, 12, 160)  179200      activation_244[0][0]             
    __________________________________________________________________________________________________
    conv2d_259 (Conv2D)             (None, 12, 12, 160)  179200      activation_249[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_245 (BatchN (None, 12, 12, 160)  480         conv2d_254[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_250 (BatchN (None, 12, 12, 160)  480         conv2d_259[0][0]                 
    __________________________________________________________________________________________________
    activation_245 (Activation)     (None, 12, 12, 160)  0           batch_normalization_245[0][0]    
    __________________________________________________________________________________________________
    activation_250 (Activation)     (None, 12, 12, 160)  0           batch_normalization_250[0][0]    
    __________________________________________________________________________________________________
    average_pooling2d_5 (AveragePoo (None, 12, 12, 768)  0           mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_252 (Conv2D)             (None, 12, 12, 192)  147456      mixed4[0][0]                     
    __________________________________________________________________________________________________
    conv2d_255 (Conv2D)             (None, 12, 12, 192)  215040      activation_245[0][0]             
    __________________________________________________________________________________________________
    conv2d_260 (Conv2D)             (None, 12, 12, 192)  215040      activation_250[0][0]             
    __________________________________________________________________________________________________
    conv2d_261 (Conv2D)             (None, 12, 12, 192)  147456      average_pooling2d_5[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_243 (BatchN (None, 12, 12, 192)  576         conv2d_252[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_246 (BatchN (None, 12, 12, 192)  576         conv2d_255[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_251 (BatchN (None, 12, 12, 192)  576         conv2d_260[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_252 (BatchN (None, 12, 12, 192)  576         conv2d_261[0][0]                 
    __________________________________________________________________________________________________
    activation_243 (Activation)     (None, 12, 12, 192)  0           batch_normalization_243[0][0]    
    __________________________________________________________________________________________________
    activation_246 (Activation)     (None, 12, 12, 192)  0           batch_normalization_246[0][0]    
    __________________________________________________________________________________________________
    activation_251 (Activation)     (None, 12, 12, 192)  0           batch_normalization_251[0][0]    
    __________________________________________________________________________________________________
    activation_252 (Activation)     (None, 12, 12, 192)  0           batch_normalization_252[0][0]    
    __________________________________________________________________________________________________
    mixed5 (Concatenate)            (None, 12, 12, 768)  0           activation_243[0][0]             
                                                                     activation_246[0][0]             
                                                                     activation_251[0][0]             
                                                                     activation_252[0][0]             
    __________________________________________________________________________________________________
    conv2d_266 (Conv2D)             (None, 12, 12, 160)  122880      mixed5[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_257 (BatchN (None, 12, 12, 160)  480         conv2d_266[0][0]                 
    __________________________________________________________________________________________________
    activation_257 (Activation)     (None, 12, 12, 160)  0           batch_normalization_257[0][0]    
    __________________________________________________________________________________________________
    conv2d_267 (Conv2D)             (None, 12, 12, 160)  179200      activation_257[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_258 (BatchN (None, 12, 12, 160)  480         conv2d_267[0][0]                 
    __________________________________________________________________________________________________
    activation_258 (Activation)     (None, 12, 12, 160)  0           batch_normalization_258[0][0]    
    __________________________________________________________________________________________________
    conv2d_263 (Conv2D)             (None, 12, 12, 160)  122880      mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_268 (Conv2D)             (None, 12, 12, 160)  179200      activation_258[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_254 (BatchN (None, 12, 12, 160)  480         conv2d_263[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_259 (BatchN (None, 12, 12, 160)  480         conv2d_268[0][0]                 
    __________________________________________________________________________________________________
    activation_254 (Activation)     (None, 12, 12, 160)  0           batch_normalization_254[0][0]    
    __________________________________________________________________________________________________
    activation_259 (Activation)     (None, 12, 12, 160)  0           batch_normalization_259[0][0]    
    __________________________________________________________________________________________________
    conv2d_264 (Conv2D)             (None, 12, 12, 160)  179200      activation_254[0][0]             
    __________________________________________________________________________________________________
    conv2d_269 (Conv2D)             (None, 12, 12, 160)  179200      activation_259[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_255 (BatchN (None, 12, 12, 160)  480         conv2d_264[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_260 (BatchN (None, 12, 12, 160)  480         conv2d_269[0][0]                 
    __________________________________________________________________________________________________
    activation_255 (Activation)     (None, 12, 12, 160)  0           batch_normalization_255[0][0]    
    __________________________________________________________________________________________________
    activation_260 (Activation)     (None, 12, 12, 160)  0           batch_normalization_260[0][0]    
    __________________________________________________________________________________________________
    average_pooling2d_6 (AveragePoo (None, 12, 12, 768)  0           mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_262 (Conv2D)             (None, 12, 12, 192)  147456      mixed5[0][0]                     
    __________________________________________________________________________________________________
    conv2d_265 (Conv2D)             (None, 12, 12, 192)  215040      activation_255[0][0]             
    __________________________________________________________________________________________________
    conv2d_270 (Conv2D)             (None, 12, 12, 192)  215040      activation_260[0][0]             
    __________________________________________________________________________________________________
    conv2d_271 (Conv2D)             (None, 12, 12, 192)  147456      average_pooling2d_6[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_253 (BatchN (None, 12, 12, 192)  576         conv2d_262[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_256 (BatchN (None, 12, 12, 192)  576         conv2d_265[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_261 (BatchN (None, 12, 12, 192)  576         conv2d_270[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_262 (BatchN (None, 12, 12, 192)  576         conv2d_271[0][0]                 
    __________________________________________________________________________________________________
    activation_253 (Activation)     (None, 12, 12, 192)  0           batch_normalization_253[0][0]    
    __________________________________________________________________________________________________
    activation_256 (Activation)     (None, 12, 12, 192)  0           batch_normalization_256[0][0]    
    __________________________________________________________________________________________________
    activation_261 (Activation)     (None, 12, 12, 192)  0           batch_normalization_261[0][0]    
    __________________________________________________________________________________________________
    activation_262 (Activation)     (None, 12, 12, 192)  0           batch_normalization_262[0][0]    
    __________________________________________________________________________________________________
    mixed6 (Concatenate)            (None, 12, 12, 768)  0           activation_253[0][0]             
                                                                     activation_256[0][0]             
                                                                     activation_261[0][0]             
                                                                     activation_262[0][0]             
    __________________________________________________________________________________________________
    conv2d_276 (Conv2D)             (None, 12, 12, 192)  147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_267 (BatchN (None, 12, 12, 192)  576         conv2d_276[0][0]                 
    __________________________________________________________________________________________________
    activation_267 (Activation)     (None, 12, 12, 192)  0           batch_normalization_267[0][0]    
    __________________________________________________________________________________________________
    conv2d_277 (Conv2D)             (None, 12, 12, 192)  258048      activation_267[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_268 (BatchN (None, 12, 12, 192)  576         conv2d_277[0][0]                 
    __________________________________________________________________________________________________
    activation_268 (Activation)     (None, 12, 12, 192)  0           batch_normalization_268[0][0]    
    __________________________________________________________________________________________________
    conv2d_273 (Conv2D)             (None, 12, 12, 192)  147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_278 (Conv2D)             (None, 12, 12, 192)  258048      activation_268[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_264 (BatchN (None, 12, 12, 192)  576         conv2d_273[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_269 (BatchN (None, 12, 12, 192)  576         conv2d_278[0][0]                 
    __________________________________________________________________________________________________
    activation_264 (Activation)     (None, 12, 12, 192)  0           batch_normalization_264[0][0]    
    __________________________________________________________________________________________________
    activation_269 (Activation)     (None, 12, 12, 192)  0           batch_normalization_269[0][0]    
    __________________________________________________________________________________________________
    conv2d_274 (Conv2D)             (None, 12, 12, 192)  258048      activation_264[0][0]             
    __________________________________________________________________________________________________
    conv2d_279 (Conv2D)             (None, 12, 12, 192)  258048      activation_269[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_265 (BatchN (None, 12, 12, 192)  576         conv2d_274[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_270 (BatchN (None, 12, 12, 192)  576         conv2d_279[0][0]                 
    __________________________________________________________________________________________________
    activation_265 (Activation)     (None, 12, 12, 192)  0           batch_normalization_265[0][0]    
    __________________________________________________________________________________________________
    activation_270 (Activation)     (None, 12, 12, 192)  0           batch_normalization_270[0][0]    
    __________________________________________________________________________________________________
    average_pooling2d_7 (AveragePoo (None, 12, 12, 768)  0           mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_272 (Conv2D)             (None, 12, 12, 192)  147456      mixed6[0][0]                     
    __________________________________________________________________________________________________
    conv2d_275 (Conv2D)             (None, 12, 12, 192)  258048      activation_265[0][0]             
    __________________________________________________________________________________________________
    conv2d_280 (Conv2D)             (None, 12, 12, 192)  258048      activation_270[0][0]             
    __________________________________________________________________________________________________
    conv2d_281 (Conv2D)             (None, 12, 12, 192)  147456      average_pooling2d_7[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_263 (BatchN (None, 12, 12, 192)  576         conv2d_272[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_266 (BatchN (None, 12, 12, 192)  576         conv2d_275[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_271 (BatchN (None, 12, 12, 192)  576         conv2d_280[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_272 (BatchN (None, 12, 12, 192)  576         conv2d_281[0][0]                 
    __________________________________________________________________________________________________
    activation_263 (Activation)     (None, 12, 12, 192)  0           batch_normalization_263[0][0]    
    __________________________________________________________________________________________________
    activation_266 (Activation)     (None, 12, 12, 192)  0           batch_normalization_266[0][0]    
    __________________________________________________________________________________________________
    activation_271 (Activation)     (None, 12, 12, 192)  0           batch_normalization_271[0][0]    
    __________________________________________________________________________________________________
    activation_272 (Activation)     (None, 12, 12, 192)  0           batch_normalization_272[0][0]    
    __________________________________________________________________________________________________
    mixed7 (Concatenate)            (None, 12, 12, 768)  0           activation_263[0][0]             
                                                                     activation_266[0][0]             
                                                                     activation_271[0][0]             
                                                                     activation_272[0][0]             
    __________________________________________________________________________________________________
    conv2d_284 (Conv2D)             (None, 12, 12, 192)  147456      mixed7[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_275 (BatchN (None, 12, 12, 192)  576         conv2d_284[0][0]                 
    __________________________________________________________________________________________________
    activation_275 (Activation)     (None, 12, 12, 192)  0           batch_normalization_275[0][0]    
    __________________________________________________________________________________________________
    conv2d_285 (Conv2D)             (None, 12, 12, 192)  258048      activation_275[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_276 (BatchN (None, 12, 12, 192)  576         conv2d_285[0][0]                 
    __________________________________________________________________________________________________
    activation_276 (Activation)     (None, 12, 12, 192)  0           batch_normalization_276[0][0]    
    __________________________________________________________________________________________________
    conv2d_282 (Conv2D)             (None, 12, 12, 192)  147456      mixed7[0][0]                     
    __________________________________________________________________________________________________
    conv2d_286 (Conv2D)             (None, 12, 12, 192)  258048      activation_276[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_273 (BatchN (None, 12, 12, 192)  576         conv2d_282[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_277 (BatchN (None, 12, 12, 192)  576         conv2d_286[0][0]                 
    __________________________________________________________________________________________________
    activation_273 (Activation)     (None, 12, 12, 192)  0           batch_normalization_273[0][0]    
    __________________________________________________________________________________________________
    activation_277 (Activation)     (None, 12, 12, 192)  0           batch_normalization_277[0][0]    
    __________________________________________________________________________________________________
    conv2d_283 (Conv2D)             (None, 5, 5, 320)    552960      activation_273[0][0]             
    __________________________________________________________________________________________________
    conv2d_287 (Conv2D)             (None, 5, 5, 192)    331776      activation_277[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_274 (BatchN (None, 5, 5, 320)    960         conv2d_283[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_278 (BatchN (None, 5, 5, 192)    576         conv2d_287[0][0]                 
    __________________________________________________________________________________________________
    activation_274 (Activation)     (None, 5, 5, 320)    0           batch_normalization_274[0][0]    
    __________________________________________________________________________________________________
    activation_278 (Activation)     (None, 5, 5, 192)    0           batch_normalization_278[0][0]    
    __________________________________________________________________________________________________
    max_pooling2d_10 (MaxPooling2D) (None, 5, 5, 768)    0           mixed7[0][0]                     
    __________________________________________________________________________________________________
    mixed8 (Concatenate)            (None, 5, 5, 1280)   0           activation_274[0][0]             
                                                                     activation_278[0][0]             
                                                                     max_pooling2d_10[0][0]           
    __________________________________________________________________________________________________
    conv2d_292 (Conv2D)             (None, 5, 5, 448)    573440      mixed8[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_283 (BatchN (None, 5, 5, 448)    1344        conv2d_292[0][0]                 
    __________________________________________________________________________________________________
    activation_283 (Activation)     (None, 5, 5, 448)    0           batch_normalization_283[0][0]    
    __________________________________________________________________________________________________
    conv2d_289 (Conv2D)             (None, 5, 5, 384)    491520      mixed8[0][0]                     
    __________________________________________________________________________________________________
    conv2d_293 (Conv2D)             (None, 5, 5, 384)    1548288     activation_283[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_280 (BatchN (None, 5, 5, 384)    1152        conv2d_289[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_284 (BatchN (None, 5, 5, 384)    1152        conv2d_293[0][0]                 
    __________________________________________________________________________________________________
    activation_280 (Activation)     (None, 5, 5, 384)    0           batch_normalization_280[0][0]    
    __________________________________________________________________________________________________
    activation_284 (Activation)     (None, 5, 5, 384)    0           batch_normalization_284[0][0]    
    __________________________________________________________________________________________________
    conv2d_290 (Conv2D)             (None, 5, 5, 384)    442368      activation_280[0][0]             
    __________________________________________________________________________________________________
    conv2d_291 (Conv2D)             (None, 5, 5, 384)    442368      activation_280[0][0]             
    __________________________________________________________________________________________________
    conv2d_294 (Conv2D)             (None, 5, 5, 384)    442368      activation_284[0][0]             
    __________________________________________________________________________________________________
    conv2d_295 (Conv2D)             (None, 5, 5, 384)    442368      activation_284[0][0]             
    __________________________________________________________________________________________________
    average_pooling2d_8 (AveragePoo (None, 5, 5, 1280)   0           mixed8[0][0]                     
    __________________________________________________________________________________________________
    conv2d_288 (Conv2D)             (None, 5, 5, 320)    409600      mixed8[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_281 (BatchN (None, 5, 5, 384)    1152        conv2d_290[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_282 (BatchN (None, 5, 5, 384)    1152        conv2d_291[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_285 (BatchN (None, 5, 5, 384)    1152        conv2d_294[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_286 (BatchN (None, 5, 5, 384)    1152        conv2d_295[0][0]                 
    __________________________________________________________________________________________________
    conv2d_296 (Conv2D)             (None, 5, 5, 192)    245760      average_pooling2d_8[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_279 (BatchN (None, 5, 5, 320)    960         conv2d_288[0][0]                 
    __________________________________________________________________________________________________
    activation_281 (Activation)     (None, 5, 5, 384)    0           batch_normalization_281[0][0]    
    __________________________________________________________________________________________________
    activation_282 (Activation)     (None, 5, 5, 384)    0           batch_normalization_282[0][0]    
    __________________________________________________________________________________________________
    activation_285 (Activation)     (None, 5, 5, 384)    0           batch_normalization_285[0][0]    
    __________________________________________________________________________________________________
    activation_286 (Activation)     (None, 5, 5, 384)    0           batch_normalization_286[0][0]    
    __________________________________________________________________________________________________
    batch_normalization_287 (BatchN (None, 5, 5, 192)    576         conv2d_296[0][0]                 
    __________________________________________________________________________________________________
    activation_279 (Activation)     (None, 5, 5, 320)    0           batch_normalization_279[0][0]    
    __________________________________________________________________________________________________
    mixed9_0 (Concatenate)          (None, 5, 5, 768)    0           activation_281[0][0]             
                                                                     activation_282[0][0]             
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 5, 5, 768)    0           activation_285[0][0]             
                                                                     activation_286[0][0]             
    __________________________________________________________________________________________________
    activation_287 (Activation)     (None, 5, 5, 192)    0           batch_normalization_287[0][0]    
    __________________________________________________________________________________________________
    mixed9 (Concatenate)            (None, 5, 5, 2048)   0           activation_279[0][0]             
                                                                     mixed9_0[0][0]                   
                                                                     concatenate[0][0]                
                                                                     activation_287[0][0]             
    __________________________________________________________________________________________________
    conv2d_301 (Conv2D)             (None, 5, 5, 448)    917504      mixed9[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_292 (BatchN (None, 5, 5, 448)    1344        conv2d_301[0][0]                 
    __________________________________________________________________________________________________
    activation_292 (Activation)     (None, 5, 5, 448)    0           batch_normalization_292[0][0]    
    __________________________________________________________________________________________________
    conv2d_298 (Conv2D)             (None, 5, 5, 384)    786432      mixed9[0][0]                     
    __________________________________________________________________________________________________
    conv2d_302 (Conv2D)             (None, 5, 5, 384)    1548288     activation_292[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_289 (BatchN (None, 5, 5, 384)    1152        conv2d_298[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_293 (BatchN (None, 5, 5, 384)    1152        conv2d_302[0][0]                 
    __________________________________________________________________________________________________
    activation_289 (Activation)     (None, 5, 5, 384)    0           batch_normalization_289[0][0]    
    __________________________________________________________________________________________________
    activation_293 (Activation)     (None, 5, 5, 384)    0           batch_normalization_293[0][0]    
    __________________________________________________________________________________________________
    conv2d_299 (Conv2D)             (None, 5, 5, 384)    442368      activation_289[0][0]             
    __________________________________________________________________________________________________
    conv2d_300 (Conv2D)             (None, 5, 5, 384)    442368      activation_289[0][0]             
    __________________________________________________________________________________________________
    conv2d_303 (Conv2D)             (None, 5, 5, 384)    442368      activation_293[0][0]             
    __________________________________________________________________________________________________
    conv2d_304 (Conv2D)             (None, 5, 5, 384)    442368      activation_293[0][0]             
    __________________________________________________________________________________________________
    average_pooling2d_9 (AveragePoo (None, 5, 5, 2048)   0           mixed9[0][0]                     
    __________________________________________________________________________________________________
    conv2d_297 (Conv2D)             (None, 5, 5, 320)    655360      mixed9[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_290 (BatchN (None, 5, 5, 384)    1152        conv2d_299[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_291 (BatchN (None, 5, 5, 384)    1152        conv2d_300[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_294 (BatchN (None, 5, 5, 384)    1152        conv2d_303[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_295 (BatchN (None, 5, 5, 384)    1152        conv2d_304[0][0]                 
    __________________________________________________________________________________________________
    conv2d_305 (Conv2D)             (None, 5, 5, 192)    393216      average_pooling2d_9[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_288 (BatchN (None, 5, 5, 320)    960         conv2d_297[0][0]                 
    __________________________________________________________________________________________________
    activation_290 (Activation)     (None, 5, 5, 384)    0           batch_normalization_290[0][0]    
    __________________________________________________________________________________________________
    activation_291 (Activation)     (None, 5, 5, 384)    0           batch_normalization_291[0][0]    
    __________________________________________________________________________________________________
    activation_294 (Activation)     (None, 5, 5, 384)    0           batch_normalization_294[0][0]    
    __________________________________________________________________________________________________
    activation_295 (Activation)     (None, 5, 5, 384)    0           batch_normalization_295[0][0]    
    __________________________________________________________________________________________________
    batch_normalization_296 (BatchN (None, 5, 5, 192)    576         conv2d_305[0][0]                 
    __________________________________________________________________________________________________
    activation_288 (Activation)     (None, 5, 5, 320)    0           batch_normalization_288[0][0]    
    __________________________________________________________________________________________________
    mixed9_1 (Concatenate)          (None, 5, 5, 768)    0           activation_290[0][0]             
                                                                     activation_291[0][0]             
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 5, 5, 768)    0           activation_294[0][0]             
                                                                     activation_295[0][0]             
    __________________________________________________________________________________________________
    activation_296 (Activation)     (None, 5, 5, 192)    0           batch_normalization_296[0][0]    
    __________________________________________________________________________________________________
    mixed10 (Concatenate)           (None, 5, 5, 2048)   0           activation_288[0][0]             
                                                                     mixed9_1[0][0]                   
                                                                     concatenate_1[0][0]              
                                                                     activation_296[0][0]             
    __________________________________________________________________________________________________
    flatten_3 (Flatten)             (None, 51200)        0           mixed10[0][0]                    
    __________________________________________________________________________________________________
    dense_9 (Dense)                 (None, 4096)         209719296   flatten_3[0][0]                  
    __________________________________________________________________________________________________
    re_lu_6 (ReLU)                  (None, 4096)         0           dense_9[0][0]                    
    __________________________________________________________________________________________________
    dropout_9 (Dropout)             (None, 4096)         0           re_lu_6[0][0]                    
    __________________________________________________________________________________________________
    dense_10 (Dense)                (None, 4096)         16781312    dropout_9[0][0]                  
    __________________________________________________________________________________________________
    re_lu_7 (ReLU)                  (None, 4096)         0           dense_10[0][0]                   
    __________________________________________________________________________________________________
    dropout_10 (Dropout)            (None, 4096)         0           re_lu_7[0][0]                    
    __________________________________________________________________________________________________
    dense_11 (Dense)                (None, 3)            12291       dropout_10[0][0]                 
    ==================================================================================================
    Total params: 248,315,683
    Trainable params: 226,512,899
    Non-trainable params: 21,802,784
    __________________________________________________________________________________________________
    None
    Epoch 1/20
    142/142 [==============================] - 41s 253ms/step - loss: 3.1483 - accuracy: 0.6148 - auc: 0.8732 - cohen_kappa: 0.6308 - f1_score: 0.7463 - precision: 0.7666 - recall: 0.7580 - val_loss: 0.5908 - val_accuracy: 0.8866 - val_auc: 0.9554 - val_cohen_kappa: 0.8239 - val_f1_score: 0.8875 - val_precision: 0.8884 - val_recall: 0.8866
    Epoch 2/20
    142/142 [==============================] - 38s 269ms/step - loss: 0.8758 - accuracy: 0.8419 - auc: 0.9328 - cohen_kappa: 0.7507 - f1_score: 0.8261 - precision: 0.8452 - recall: 0.8395 - val_loss: 0.2780 - val_accuracy: 0.9155 - val_auc: 0.9793 - val_cohen_kappa: 0.8705 - val_f1_score: 0.9128 - val_precision: 0.9210 - val_recall: 0.9134
    Epoch 3/20
    142/142 [==============================] - 34s 238ms/step - loss: 0.3148 - accuracy: 0.9090 - auc: 0.9748 - cohen_kappa: 0.8577 - f1_score: 0.8972 - precision: 0.9124 - recall: 0.9055 - val_loss: 0.1424 - val_accuracy: 0.9464 - val_auc: 0.9942 - val_cohen_kappa: 0.9174 - val_f1_score: 0.9461 - val_precision: 0.9520 - val_recall: 0.9402
    Epoch 4/20
    142/142 [==============================] - 38s 267ms/step - loss: 0.2064 - accuracy: 0.9278 - auc: 0.9878 - cohen_kappa: 0.8867 - f1_score: 0.9218 - precision: 0.9347 - recall: 0.9229 - val_loss: 0.2567 - val_accuracy: 0.9216 - val_auc: 0.9849 - val_cohen_kappa: 0.8794 - val_f1_score: 0.9227 - val_precision: 0.9252 - val_recall: 0.9175
    Epoch 5/20
    142/142 [==============================] - 34s 237ms/step - loss: 0.2242 - accuracy: 0.9293 - auc: 0.9860 - cohen_kappa: 0.8891 - f1_score: 0.9245 - precision: 0.9315 - recall: 0.9255 - val_loss: 0.1649 - val_accuracy: 0.9340 - val_auc: 0.9930 - val_cohen_kappa: 0.8979 - val_f1_score: 0.9304 - val_precision: 0.9433 - val_recall: 0.9258
    Epoch 6/20
    142/142 [==============================] - 38s 268ms/step - loss: 0.1398 - accuracy: 0.9514 - auc: 0.9935 - cohen_kappa: 0.9242 - f1_score: 0.9468 - precision: 0.9539 - recall: 0.9463 - val_loss: 0.0971 - val_accuracy: 0.9670 - val_auc: 0.9974 - val_cohen_kappa: 0.9495 - val_f1_score: 0.9681 - val_precision: 0.9690 - val_recall: 0.9670
    Epoch 7/20
    142/142 [==============================] - 34s 236ms/step - loss: 0.1161 - accuracy: 0.9608 - auc: 0.9955 - cohen_kappa: 0.9385 - f1_score: 0.9578 - precision: 0.9654 - recall: 0.9595 - val_loss: 0.3148 - val_accuracy: 0.8928 - val_auc: 0.9773 - val_cohen_kappa: 0.8354 - val_f1_score: 0.8920 - val_precision: 0.9070 - val_recall: 0.8845
    Epoch 8/20
    142/142 [==============================] - 38s 269ms/step - loss: 0.1668 - accuracy: 0.9379 - auc: 0.9912 - cohen_kappa: 0.9029 - f1_score: 0.9361 - precision: 0.9422 - recall: 0.9355 - val_loss: 0.1477 - val_accuracy: 0.9505 - val_auc: 0.9939 - val_cohen_kappa: 0.9240 - val_f1_score: 0.9525 - val_precision: 0.9503 - val_recall: 0.9464
    Epoch 9/20
    142/142 [==============================] - 34s 238ms/step - loss: 0.1182 - accuracy: 0.9590 - auc: 0.9950 - cohen_kappa: 0.9362 - f1_score: 0.9549 - precision: 0.9609 - recall: 0.9585 - val_loss: 0.1030 - val_accuracy: 0.9649 - val_auc: 0.9959 - val_cohen_kappa: 0.9462 - val_f1_score: 0.9652 - val_precision: 0.9649 - val_recall: 0.9649
    Epoch 10/20
    142/142 [==============================] - 38s 267ms/step - loss: 0.1254 - accuracy: 0.9573 - auc: 0.9946 - cohen_kappa: 0.9330 - f1_score: 0.9545 - precision: 0.9590 - recall: 0.9572 - val_loss: 0.1217 - val_accuracy: 0.9526 - val_auc: 0.9956 - val_cohen_kappa: 0.9271 - val_f1_score: 0.9522 - val_precision: 0.9545 - val_recall: 0.9505
    Epoch 11/20
    142/142 [==============================] - 34s 239ms/step - loss: 0.1015 - accuracy: 0.9628 - auc: 0.9961 - cohen_kappa: 0.9413 - f1_score: 0.9596 - precision: 0.9660 - recall: 0.9599 - val_loss: 0.0650 - val_accuracy: 0.9753 - val_auc: 0.9988 - val_cohen_kappa: 0.9621 - val_f1_score: 0.9756 - val_precision: 0.9752 - val_recall: 0.9732
    Epoch 12/20
    142/142 [==============================] - 38s 267ms/step - loss: 0.1151 - accuracy: 0.9587 - auc: 0.9956 - cohen_kappa: 0.9358 - f1_score: 0.9554 - precision: 0.9599 - recall: 0.9582 - val_loss: 0.0850 - val_accuracy: 0.9588 - val_auc: 0.9981 - val_cohen_kappa: 0.9367 - val_f1_score: 0.9587 - val_precision: 0.9607 - val_recall: 0.9588
    Epoch 13/20
    142/142 [==============================] - 34s 241ms/step - loss: 0.0995 - accuracy: 0.9620 - auc: 0.9968 - cohen_kappa: 0.9410 - f1_score: 0.9594 - precision: 0.9643 - recall: 0.9617 - val_loss: 0.0678 - val_accuracy: 0.9711 - val_auc: 0.9987 - val_cohen_kappa: 0.9556 - val_f1_score: 0.9709 - val_precision: 0.9731 - val_recall: 0.9711
    Epoch 14/20
    142/142 [==============================] - 38s 266ms/step - loss: 0.0806 - accuracy: 0.9712 - auc: 0.9972 - cohen_kappa: 0.9549 - f1_score: 0.9692 - precision: 0.9751 - recall: 0.9679 - val_loss: 0.1077 - val_accuracy: 0.9567 - val_auc: 0.9960 - val_cohen_kappa: 0.9338 - val_f1_score: 0.9547 - val_precision: 0.9607 - val_recall: 0.9567
    Epoch 15/20
    142/142 [==============================] - 34s 238ms/step - loss: 0.1057 - accuracy: 0.9679 - auc: 0.9952 - cohen_kappa: 0.9492 - f1_score: 0.9652 - precision: 0.9689 - recall: 0.9645 - val_loss: 0.0635 - val_accuracy: 0.9753 - val_auc: 0.9988 - val_cohen_kappa: 0.9621 - val_f1_score: 0.9749 - val_precision: 0.9752 - val_recall: 0.9732
    Epoch 16/20
    142/142 [==============================] - 38s 269ms/step - loss: 0.0809 - accuracy: 0.9735 - auc: 0.9978 - cohen_kappa: 0.9583 - f1_score: 0.9699 - precision: 0.9748 - recall: 0.9722 - val_loss: 0.0748 - val_accuracy: 0.9711 - val_auc: 0.9984 - val_cohen_kappa: 0.9557 - val_f1_score: 0.9709 - val_precision: 0.9731 - val_recall: 0.9691
    Epoch 17/20
    142/142 [==============================] - 34s 236ms/step - loss: 0.0661 - accuracy: 0.9807 - auc: 0.9980 - cohen_kappa: 0.9698 - f1_score: 0.9797 - precision: 0.9836 - recall: 0.9785 - val_loss: 0.0857 - val_accuracy: 0.9649 - val_auc: 0.9978 - val_cohen_kappa: 0.9461 - val_f1_score: 0.9664 - val_precision: 0.9688 - val_recall: 0.9608
    Epoch 18/20
    142/142 [==============================] - 38s 270ms/step - loss: 0.0691 - accuracy: 0.9725 - auc: 0.9985 - cohen_kappa: 0.9571 - f1_score: 0.9697 - precision: 0.9744 - recall: 0.9721 - val_loss: 0.0489 - val_accuracy: 0.9814 - val_auc: 0.9993 - val_cohen_kappa: 0.9716 - val_f1_score: 0.9810 - val_precision: 0.9814 - val_recall: 0.9814
    Epoch 19/20
    142/142 [==============================] - 34s 236ms/step - loss: 0.0588 - accuracy: 0.9768 - auc: 0.9987 - cohen_kappa: 0.9638 - f1_score: 0.9757 - precision: 0.9791 - recall: 0.9760 - val_loss: 0.0552 - val_accuracy: 0.9856 - val_auc: 0.9982 - val_cohen_kappa: 0.9779 - val_f1_score: 0.9856 - val_precision: 0.9896 - val_recall: 0.9814
    Epoch 20/20
    142/142 [==============================] - 38s 270ms/step - loss: 0.0892 - accuracy: 0.9696 - auc: 0.9973 - cohen_kappa: 0.9526 - f1_score: 0.9685 - precision: 0.9720 - recall: 0.9681 - val_loss: 0.0422 - val_accuracy: 0.9835 - val_auc: 0.9995 - val_cohen_kappa: 0.9747 - val_f1_score: 0.9839 - val_precision: 0.9855 - val_recall: 0.9835
    


```python
evaluate_model(inception_model, inception_history, test_generator)
```

    
    Test set accuracy: 0.985567033290863 
    
    31/31 [==============================] - 4s 78ms/step
    
                  precision    recall  f1-score   support
    
             AMD       1.00      0.99      1.00       121
             DME       0.99      0.96      0.98       155
          NORMAL       0.97      1.00      0.99       209
    
        accuracy                           0.99       485
       macro avg       0.99      0.98      0.99       485
    weighted avg       0.99      0.99      0.99       485
    
    


![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_34_1.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_34_2.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_34_3.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_34_4.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_34_5.png)



![png](2022-06-26-oct-duke-all-final_files/2022-06-26-oct-duke-all-final_34_6.png)


    ROC AUC score: 0.9998214443843466
    
