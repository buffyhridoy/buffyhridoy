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

    100%|██████████| 723/723 [00:00<00:00, 329259.70it/s]
    100%|██████████| 1407/1407 [00:00<00:00, 322568.23it/s]
    100%|██████████| 1101/1101 [00:00<00:00, 315626.32it/s]
    


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




![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_5_1.png)



```python
sns.countplot(x = Y_val)
```




    <AxesSubplot:ylabel='count'>




![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_6_1.png)



```python
sns.countplot(x = Y_test)
```




    <AxesSubplot:ylabel='count'>




![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_7_1.png)



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
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=6, verbose=1)
    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=7)
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
    142/142 [==============================] - 56s 314ms/step - loss: 0.9654 - accuracy: 0.5721 - auc: 0.7591 - cohen_kappa: 0.3230 - f1_score: 0.5350 - precision: 0.6144 - recall: 0.5215 - val_loss: 0.1594 - val_accuracy: 0.9485 - val_auc: 0.9923 - val_cohen_kappa: 0.9208 - val_f1_score: 0.9496 - val_precision: 0.9499 - val_recall: 0.9381
    Epoch 2/50
    142/142 [==============================] - 37s 259ms/step - loss: 0.1467 - accuracy: 0.9480 - auc: 0.9934 - cohen_kappa: 0.9186 - f1_score: 0.9444 - precision: 0.9519 - recall: 0.9462 - val_loss: 0.0662 - val_accuracy: 0.9732 - val_auc: 0.9988 - val_cohen_kappa: 0.9588 - val_f1_score: 0.9727 - val_precision: 0.9752 - val_recall: 0.9732
    Epoch 3/50
    142/142 [==============================] - 35s 250ms/step - loss: 0.0920 - accuracy: 0.9708 - auc: 0.9968 - cohen_kappa: 0.9536 - f1_score: 0.9668 - precision: 0.9744 - recall: 0.9700 - val_loss: 0.1033 - val_accuracy: 0.9629 - val_auc: 0.9969 - val_cohen_kappa: 0.9429 - val_f1_score: 0.9617 - val_precision: 0.9648 - val_recall: 0.9608
    Epoch 4/50
    142/142 [==============================] - 36s 251ms/step - loss: 0.0543 - accuracy: 0.9779 - auc: 0.9990 - cohen_kappa: 0.9650 - f1_score: 0.9736 - precision: 0.9780 - recall: 0.9764 - val_loss: 0.0726 - val_accuracy: 0.9732 - val_auc: 0.9988 - val_cohen_kappa: 0.9589 - val_f1_score: 0.9732 - val_precision: 0.9731 - val_recall: 0.9711
    Epoch 5/50
    142/142 [==============================] - 36s 252ms/step - loss: 0.0644 - accuracy: 0.9766 - auc: 0.9983 - cohen_kappa: 0.9635 - f1_score: 0.9766 - precision: 0.9766 - recall: 0.9766 - val_loss: 0.0839 - val_accuracy: 0.9732 - val_auc: 0.9965 - val_cohen_kappa: 0.9588 - val_f1_score: 0.9756 - val_precision: 0.9752 - val_recall: 0.9732
    Epoch 6/50
    142/142 [==============================] - 36s 257ms/step - loss: 0.0358 - accuracy: 0.9902 - auc: 0.9991 - cohen_kappa: 0.9847 - f1_score: 0.9897 - precision: 0.9902 - recall: 0.9896 - val_loss: 0.1885 - val_accuracy: 0.9443 - val_auc: 0.9876 - val_cohen_kappa: 0.9144 - val_f1_score: 0.9487 - val_precision: 0.9442 - val_recall: 0.9423
    Epoch 7/50
    142/142 [==============================] - 36s 253ms/step - loss: 0.0458 - accuracy: 0.9837 - auc: 0.9979 - cohen_kappa: 0.9744 - f1_score: 0.9833 - precision: 0.9852 - recall: 0.9837 - val_loss: 0.0119 - val_accuracy: 0.9938 - val_auc: 1.0000 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9939 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 8/50
    142/142 [==============================] - 38s 265ms/step - loss: 0.0267 - accuracy: 0.9922 - auc: 0.9990 - cohen_kappa: 0.9877 - f1_score: 0.9931 - precision: 0.9922 - recall: 0.9919 - val_loss: 0.0289 - val_accuracy: 0.9938 - val_auc: 0.9983 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9945 - val_precision: 0.9938 - val_recall: 0.9918
    Epoch 9/50
    142/142 [==============================] - 37s 258ms/step - loss: 0.0207 - accuracy: 0.9951 - auc: 0.9991 - cohen_kappa: 0.9924 - f1_score: 0.9954 - precision: 0.9952 - recall: 0.9951 - val_loss: 0.0273 - val_accuracy: 0.9876 - val_auc: 0.9998 - val_cohen_kappa: 0.9810 - val_f1_score: 0.9867 - val_precision: 0.9876 - val_recall: 0.9876
    Epoch 10/50
    142/142 [==============================] - 36s 253ms/step - loss: 0.0170 - accuracy: 0.9938 - auc: 0.9999 - cohen_kappa: 0.9904 - f1_score: 0.9946 - precision: 0.9942 - recall: 0.9938 - val_loss: 0.0739 - val_accuracy: 0.9711 - val_auc: 0.9974 - val_cohen_kappa: 0.9557 - val_f1_score: 0.9737 - val_precision: 0.9711 - val_recall: 0.9711
    Epoch 11/50
    142/142 [==============================] - 37s 262ms/step - loss: 0.0141 - accuracy: 0.9972 - auc: 0.9999 - cohen_kappa: 0.9957 - f1_score: 0.9973 - precision: 0.9972 - recall: 0.9972 - val_loss: 0.0611 - val_accuracy: 0.9814 - val_auc: 0.9963 - val_cohen_kappa: 0.9715 - val_f1_score: 0.9835 - val_precision: 0.9814 - val_recall: 0.9814
    Epoch 12/50
    142/142 [==============================] - 37s 257ms/step - loss: 0.0112 - accuracy: 0.9949 - auc: 1.0000 - cohen_kappa: 0.9920 - f1_score: 0.9946 - precision: 0.9961 - recall: 0.9949 - val_loss: 0.0074 - val_accuracy: 1.0000 - val_auc: 1.0000 - val_cohen_kappa: 1.0000 - val_f1_score: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
    Epoch 13/50
    142/142 [==============================] - 37s 256ms/step - loss: 0.0164 - accuracy: 0.9961 - auc: 0.9995 - cohen_kappa: 0.9938 - f1_score: 0.9966 - precision: 0.9963 - recall: 0.9961 - val_loss: 0.0250 - val_accuracy: 0.9938 - val_auc: 0.9998 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9945 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 14/50
    142/142 [==============================] - 36s 251ms/step - loss: 0.0221 - accuracy: 0.9895 - auc: 0.9998 - cohen_kappa: 0.9834 - f1_score: 0.9876 - precision: 0.9895 - recall: 0.9895 - val_loss: 0.0316 - val_accuracy: 0.9856 - val_auc: 0.9997 - val_cohen_kappa: 0.9779 - val_f1_score: 0.9855 - val_precision: 0.9856 - val_recall: 0.9856
    Epoch 15/50
    142/142 [==============================] - 36s 251ms/step - loss: 0.0135 - accuracy: 0.9954 - auc: 0.9999 - cohen_kappa: 0.9929 - f1_score: 0.9959 - precision: 0.9954 - recall: 0.9954 - val_loss: 0.0031 - val_accuracy: 1.0000 - val_auc: 1.0000 - val_cohen_kappa: 1.0000 - val_f1_score: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
    Epoch 16/50
    142/142 [==============================] - 36s 254ms/step - loss: 0.0111 - accuracy: 0.9969 - auc: 1.0000 - cohen_kappa: 0.9952 - f1_score: 0.9968 - precision: 0.9969 - recall: 0.9969 - val_loss: 0.0235 - val_accuracy: 0.9938 - val_auc: 0.9998 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9939 - val_precision: 0.9938 - val_recall: 0.9918
    Epoch 17/50
    142/142 [==============================] - 36s 253ms/step - loss: 0.0086 - accuracy: 0.9976 - auc: 1.0000 - cohen_kappa: 0.9962 - f1_score: 0.9956 - precision: 0.9976 - recall: 0.9970 - val_loss: 0.0102 - val_accuracy: 0.9959 - val_auc: 1.0000 - val_cohen_kappa: 0.9937 - val_f1_score: 0.9964 - val_precision: 0.9959 - val_recall: 0.9959
    Epoch 18/50
    142/142 [==============================] - 36s 252ms/step - loss: 0.0028 - accuracy: 0.9996 - auc: 1.0000 - cohen_kappa: 0.9993 - f1_score: 0.9995 - precision: 0.9996 - recall: 0.9996 - val_loss: 0.0044 - val_accuracy: 0.9979 - val_auc: 1.0000 - val_cohen_kappa: 0.9968 - val_f1_score: 0.9982 - val_precision: 0.9979 - val_recall: 0.9979
    Epoch 00018: early stopping
    


```python
evaluate_model(vgg_model, vgg_history, test_generator)
```

    
    Test set accuracy: 1.0 
    
    31/31 [==============================] - 3s 73ms/step
    
                  precision    recall  f1-score   support
    
             AMD       1.00      1.00      1.00       121
             DME       1.00      1.00      1.00       155
          NORMAL       1.00      1.00      1.00       209
    
        accuracy                           1.00       485
       macro avg       1.00      1.00      1.00       485
    weighted avg       1.00      1.00      1.00       485
    
    


![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_16_1.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_16_2.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_16_3.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_16_4.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_16_5.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_16_6.png)


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


![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_20_0.png)



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


![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_22_0.png)



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


![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_24_0.png)



```python
incres_model = generate_model('inceptionresnet', 3)

```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/inception_resnet_v2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5
    219062272/219055592 [==============================] - 2s 0us/step
    


```python
incres_model, incresincres_history = train_model(incres_model, train_generator, val_generator, 50, tf.keras.optimizers.SGD(lr=0.001, momentum=0.9), metrics)
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
    142/142 [==============================] - 55s 304ms/step - loss: 1.0069 - accuracy: 0.6601 - auc: 0.9363 - cohen_kappa: 0.6917 - f1_score: 0.7903 - precision: 0.8178 - recall: 0.7928 - val_loss: 0.3085 - val_accuracy: 0.8907 - val_auc: 0.9762 - val_cohen_kappa: 0.8320 - val_f1_score: 0.8925 - val_precision: 0.8903 - val_recall: 0.8866
    Epoch 2/50
    142/142 [==============================] - 39s 277ms/step - loss: 0.3501 - accuracy: 0.8967 - auc: 0.9737 - cohen_kappa: 0.8365 - f1_score: 0.8888 - precision: 0.8975 - recall: 0.8950 - val_loss: 0.1110 - val_accuracy: 0.9691 - val_auc: 0.9941 - val_cohen_kappa: 0.9526 - val_f1_score: 0.9689 - val_precision: 0.9691 - val_recall: 0.9691
    Epoch 3/50
    142/142 [==============================] - 40s 277ms/step - loss: 0.2009 - accuracy: 0.9275 - auc: 0.9885 - cohen_kappa: 0.8866 - f1_score: 0.9226 - precision: 0.9276 - recall: 0.9259 - val_loss: 0.1207 - val_accuracy: 0.9567 - val_auc: 0.9942 - val_cohen_kappa: 0.9336 - val_f1_score: 0.9572 - val_precision: 0.9567 - val_recall: 0.9567
    Epoch 4/50
    142/142 [==============================] - 39s 276ms/step - loss: 0.1390 - accuracy: 0.9565 - auc: 0.9928 - cohen_kappa: 0.9318 - f1_score: 0.9535 - precision: 0.9569 - recall: 0.9533 - val_loss: 0.0521 - val_accuracy: 0.9794 - val_auc: 0.9992 - val_cohen_kappa: 0.9684 - val_f1_score: 0.9803 - val_precision: 0.9794 - val_recall: 0.9794
    Epoch 5/50
    142/142 [==============================] - 39s 277ms/step - loss: 0.1008 - accuracy: 0.9631 - auc: 0.9956 - cohen_kappa: 0.9426 - f1_score: 0.9607 - precision: 0.9653 - recall: 0.9631 - val_loss: 0.0357 - val_accuracy: 0.9835 - val_auc: 0.9997 - val_cohen_kappa: 0.9748 - val_f1_score: 0.9844 - val_precision: 0.9855 - val_recall: 0.9835
    Epoch 6/50
    142/142 [==============================] - 41s 288ms/step - loss: 0.0706 - accuracy: 0.9784 - auc: 0.9967 - cohen_kappa: 0.9662 - f1_score: 0.9780 - precision: 0.9784 - recall: 0.9784 - val_loss: 0.0191 - val_accuracy: 0.9938 - val_auc: 0.9999 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9940 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 7/50
    142/142 [==============================] - 41s 286ms/step - loss: 0.0968 - accuracy: 0.9732 - auc: 0.9951 - cohen_kappa: 0.9581 - f1_score: 0.9717 - precision: 0.9747 - recall: 0.9730 - val_loss: 0.2942 - val_accuracy: 0.9031 - val_auc: 0.9827 - val_cohen_kappa: 0.8530 - val_f1_score: 0.9089 - val_precision: 0.9066 - val_recall: 0.9010
    Epoch 8/50
    142/142 [==============================] - 39s 276ms/step - loss: 0.0918 - accuracy: 0.9663 - auc: 0.9970 - cohen_kappa: 0.9478 - f1_score: 0.9649 - precision: 0.9672 - recall: 0.9650 - val_loss: 0.0581 - val_accuracy: 0.9835 - val_auc: 0.9991 - val_cohen_kappa: 0.9747 - val_f1_score: 0.9843 - val_precision: 0.9835 - val_recall: 0.9814
    Epoch 9/50
    142/142 [==============================] - 39s 274ms/step - loss: 0.0596 - accuracy: 0.9832 - auc: 0.9979 - cohen_kappa: 0.9737 - f1_score: 0.9828 - precision: 0.9833 - recall: 0.9832 - val_loss: 0.0081 - val_accuracy: 1.0000 - val_auc: 1.0000 - val_cohen_kappa: 1.0000 - val_f1_score: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
    Epoch 10/50
    142/142 [==============================] - 39s 277ms/step - loss: 0.0336 - accuracy: 0.9894 - auc: 0.9995 - cohen_kappa: 0.9835 - f1_score: 0.9894 - precision: 0.9905 - recall: 0.9887 - val_loss: 0.0147 - val_accuracy: 0.9938 - val_auc: 0.9999 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9930 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 11/50
    142/142 [==============================] - 38s 268ms/step - loss: 0.0406 - accuracy: 0.9838 - auc: 0.9993 - cohen_kappa: 0.9747 - f1_score: 0.9824 - precision: 0.9844 - recall: 0.9838 - val_loss: 0.0313 - val_accuracy: 0.9897 - val_auc: 0.9997 - val_cohen_kappa: 0.9842 - val_f1_score: 0.9882 - val_precision: 0.9897 - val_recall: 0.9897
    Epoch 12/50
    142/142 [==============================] - 40s 278ms/step - loss: 0.0349 - accuracy: 0.9871 - auc: 0.9985 - cohen_kappa: 0.9797 - f1_score: 0.9869 - precision: 0.9883 - recall: 0.9871 - val_loss: 0.0096 - val_accuracy: 0.9959 - val_auc: 1.0000 - val_cohen_kappa: 0.9937 - val_f1_score: 0.9958 - val_precision: 0.9959 - val_recall: 0.9959
    Epoch 13/50
    142/142 [==============================] - 38s 267ms/step - loss: 0.0254 - accuracy: 0.9924 - auc: 0.9991 - cohen_kappa: 0.9882 - f1_score: 0.9918 - precision: 0.9924 - recall: 0.9924 - val_loss: 0.0069 - val_accuracy: 0.9979 - val_auc: 1.0000 - val_cohen_kappa: 0.9968 - val_f1_score: 0.9977 - val_precision: 0.9979 - val_recall: 0.9979
    Epoch 14/50
    142/142 [==============================] - 39s 275ms/step - loss: 0.0424 - accuracy: 0.9868 - auc: 0.9992 - cohen_kappa: 0.9793 - f1_score: 0.9869 - precision: 0.9868 - recall: 0.9863 - val_loss: 0.0730 - val_accuracy: 0.9814 - val_auc: 0.9964 - val_cohen_kappa: 0.9716 - val_f1_score: 0.9819 - val_precision: 0.9814 - val_recall: 0.9814
    Epoch 15/50
    142/142 [==============================] - 38s 268ms/step - loss: 0.0550 - accuracy: 0.9832 - auc: 0.9982 - cohen_kappa: 0.9736 - f1_score: 0.9830 - precision: 0.9832 - recall: 0.9832 - val_loss: 0.1000 - val_accuracy: 0.9711 - val_auc: 0.9956 - val_cohen_kappa: 0.9558 - val_f1_score: 0.9705 - val_precision: 0.9711 - val_recall: 0.9711
    Epoch 00015: early stopping
    


```python
evaluate_model(incres_model, incresincres_history, test_generator)
```

    
    Test set accuracy: 0.9896907210350037 
    
    31/31 [==============================] - 5s 81ms/step
    
                  precision    recall  f1-score   support
    
             AMD       0.98      1.00      0.99       121
             DME       1.00      0.97      0.98       155
          NORMAL       0.99      1.00      0.99       209
    
        accuracy                           0.99       485
       macro avg       0.99      0.99      0.99       485
    weighted avg       0.99      0.99      0.99       485
    
    


![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_27_1.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_27_2.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_27_3.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_27_4.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_27_5.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_27_6.png)


    ROC AUC score: 0.9998887150088019
    


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
model, history = train_model(model, train_generator, val_generator, 50, tf.keras.optimizers.SGD(lr=0.001, momentum=0.9), metrics)
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
    Epoch 1/50
    142/142 [==============================] - 47s 300ms/step - loss: 1.0803 - accuracy: 0.3864 - auc: 0.8205 - cohen_kappa: 0.3641 - f1_score: 0.5661 - precision: 0.9858 - recall: 0.3636 - val_loss: 1.0681 - val_accuracy: 0.4062 - val_auc: 0.6396 - val_cohen_kappa: 0.0000e+00 - val_f1_score: 0.1926 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/50
    142/142 [==============================] - 41s 292ms/step - loss: 1.0241 - accuracy: 0.4678 - auc: 0.6702 - cohen_kappa: 0.0730 - f1_score: 0.2805 - precision: 0.5361 - recall: 0.1183 - val_loss: 0.9874 - val_accuracy: 0.5505 - val_auc: 0.7200 - val_cohen_kappa: 0.2635 - val_f1_score: 0.4453 - val_precision: 0.5560 - val_recall: 0.2660
    Epoch 3/50
    142/142 [==============================] - 41s 290ms/step - loss: 0.9368 - accuracy: 0.5623 - auc: 0.7455 - cohen_kappa: 0.2669 - f1_score: 0.4757 - precision: 0.6058 - recall: 0.3599 - val_loss: 0.9604 - val_accuracy: 0.5711 - val_auc: 0.7574 - val_cohen_kappa: 0.3057 - val_f1_score: 0.4983 - val_precision: 0.8302 - val_recall: 0.1814
    Epoch 4/50
    142/142 [==============================] - 43s 300ms/step - loss: 0.9079 - accuracy: 0.5902 - auc: 0.7613 - cohen_kappa: 0.3254 - f1_score: 0.5411 - precision: 0.6202 - recall: 0.4172 - val_loss: 0.9228 - val_accuracy: 0.5773 - val_auc: 0.7561 - val_cohen_kappa: 0.3130 - val_f1_score: 0.4973 - val_precision: 0.5833 - val_recall: 0.5052
    Epoch 5/50
    142/142 [==============================] - 41s 288ms/step - loss: 0.8929 - accuracy: 0.5922 - auc: 0.7711 - cohen_kappa: 0.3225 - f1_score: 0.5399 - precision: 0.6334 - recall: 0.4633 - val_loss: 0.8913 - val_accuracy: 0.5959 - val_auc: 0.7721 - val_cohen_kappa: 0.3467 - val_f1_score: 0.5315 - val_precision: 0.6089 - val_recall: 0.5649
    Epoch 6/50
    142/142 [==============================] - 41s 290ms/step - loss: 0.8849 - accuracy: 0.5922 - auc: 0.7704 - cohen_kappa: 0.3389 - f1_score: 0.5524 - precision: 0.6143 - recall: 0.4607 - val_loss: 0.9281 - val_accuracy: 0.6124 - val_auc: 0.8075 - val_cohen_kappa: 0.3925 - val_f1_score: 0.5972 - val_precision: 0.8182 - val_recall: 0.1113
    Epoch 7/50
    142/142 [==============================] - 42s 294ms/step - loss: 0.8408 - accuracy: 0.6176 - auc: 0.7981 - cohen_kappa: 0.3762 - f1_score: 0.5775 - precision: 0.6540 - recall: 0.4818 - val_loss: 0.8773 - val_accuracy: 0.5856 - val_auc: 0.7812 - val_cohen_kappa: 0.3278 - val_f1_score: 0.5093 - val_precision: 0.6067 - val_recall: 0.5216
    Epoch 8/50
    142/142 [==============================] - 43s 301ms/step - loss: 0.8570 - accuracy: 0.6103 - auc: 0.7867 - cohen_kappa: 0.3578 - f1_score: 0.5644 - precision: 0.6419 - recall: 0.4958 - val_loss: 0.8631 - val_accuracy: 0.5918 - val_auc: 0.8013 - val_cohen_kappa: 0.3457 - val_f1_score: 0.5498 - val_precision: 0.7040 - val_recall: 0.4660
    Epoch 9/50
    142/142 [==============================] - 41s 291ms/step - loss: 0.8360 - accuracy: 0.6221 - auc: 0.7992 - cohen_kappa: 0.3851 - f1_score: 0.5826 - precision: 0.6548 - recall: 0.5250 - val_loss: 0.8631 - val_accuracy: 0.5835 - val_auc: 0.7889 - val_cohen_kappa: 0.3263 - val_f1_score: 0.5150 - val_precision: 0.5965 - val_recall: 0.5546
    Epoch 10/50
    142/142 [==============================] - 41s 288ms/step - loss: 0.8264 - accuracy: 0.6232 - auc: 0.8032 - cohen_kappa: 0.3960 - f1_score: 0.5933 - precision: 0.6602 - recall: 0.5289 - val_loss: 0.8323 - val_accuracy: 0.6186 - val_auc: 0.8050 - val_cohen_kappa: 0.3880 - val_f1_score: 0.5746 - val_precision: 0.6447 - val_recall: 0.5649
    Epoch 11/50
    142/142 [==============================] - 42s 296ms/step - loss: 0.8093 - accuracy: 0.6441 - auc: 0.8129 - cohen_kappa: 0.4173 - f1_score: 0.6006 - precision: 0.6763 - recall: 0.5675 - val_loss: 0.8362 - val_accuracy: 0.6103 - val_auc: 0.8034 - val_cohen_kappa: 0.3737 - val_f1_score: 0.5609 - val_precision: 0.6294 - val_recall: 0.5918
    Epoch 12/50
    142/142 [==============================] - 43s 301ms/step - loss: 0.8133 - accuracy: 0.6294 - auc: 0.8095 - cohen_kappa: 0.4009 - f1_score: 0.5980 - precision: 0.6612 - recall: 0.5361 - val_loss: 0.8098 - val_accuracy: 0.6495 - val_auc: 0.8301 - val_cohen_kappa: 0.4465 - val_f1_score: 0.6112 - val_precision: 0.7448 - val_recall: 0.4392
    Epoch 13/50
    142/142 [==============================] - 42s 292ms/step - loss: 0.8153 - accuracy: 0.6291 - auc: 0.8066 - cohen_kappa: 0.4116 - f1_score: 0.6029 - precision: 0.6710 - recall: 0.5244 - val_loss: 0.7863 - val_accuracy: 0.6186 - val_auc: 0.8310 - val_cohen_kappa: 0.3934 - val_f1_score: 0.5828 - val_precision: 0.6732 - val_recall: 0.5649
    Epoch 14/50
    142/142 [==============================] - 42s 296ms/step - loss: 0.7847 - accuracy: 0.6276 - auc: 0.8218 - cohen_kappa: 0.4039 - f1_score: 0.5990 - precision: 0.6680 - recall: 0.5377 - val_loss: 0.7615 - val_accuracy: 0.6412 - val_auc: 0.8510 - val_cohen_kappa: 0.4382 - val_f1_score: 0.6149 - val_precision: 0.7351 - val_recall: 0.5608
    Epoch 15/50
    142/142 [==============================] - 43s 300ms/step - loss: 0.7825 - accuracy: 0.6344 - auc: 0.8244 - cohen_kappa: 0.4078 - f1_score: 0.5981 - precision: 0.6803 - recall: 0.5572 - val_loss: 0.7496 - val_accuracy: 0.6722 - val_auc: 0.8666 - val_cohen_kappa: 0.4909 - val_f1_score: 0.6549 - val_precision: 0.7713 - val_recall: 0.5423
    Epoch 16/50
    142/142 [==============================] - 43s 301ms/step - loss: 0.7648 - accuracy: 0.6618 - auc: 0.8358 - cohen_kappa: 0.4541 - f1_score: 0.6311 - precision: 0.7031 - recall: 0.5765 - val_loss: 0.7384 - val_accuracy: 0.6557 - val_auc: 0.8523 - val_cohen_kappa: 0.4542 - val_f1_score: 0.6212 - val_precision: 0.6951 - val_recall: 0.5876
    Epoch 17/50
    142/142 [==============================] - 41s 288ms/step - loss: 0.7473 - accuracy: 0.6685 - auc: 0.8413 - cohen_kappa: 0.4681 - f1_score: 0.6437 - precision: 0.7101 - recall: 0.5987 - val_loss: 0.8257 - val_accuracy: 0.6000 - val_auc: 0.8228 - val_cohen_kappa: 0.3541 - val_f1_score: 0.5375 - val_precision: 0.6294 - val_recall: 0.5918
    Epoch 18/50
    142/142 [==============================] - 42s 292ms/step - loss: 0.7304 - accuracy: 0.6827 - auc: 0.8499 - cohen_kappa: 0.4787 - f1_score: 0.6467 - precision: 0.7229 - recall: 0.6141 - val_loss: 0.7206 - val_accuracy: 0.6660 - val_auc: 0.8582 - val_cohen_kappa: 0.4706 - val_f1_score: 0.6361 - val_precision: 0.7046 - val_recall: 0.6000
    Epoch 19/50
    142/142 [==============================] - 42s 295ms/step - loss: 0.7209 - accuracy: 0.6759 - auc: 0.8544 - cohen_kappa: 0.4789 - f1_score: 0.6453 - precision: 0.7061 - recall: 0.6156 - val_loss: 0.7834 - val_accuracy: 0.6330 - val_auc: 0.8439 - val_cohen_kappa: 0.4131 - val_f1_score: 0.5904 - val_precision: 0.6528 - val_recall: 0.6165
    Epoch 20/50
    142/142 [==============================] - 43s 301ms/step - loss: 0.7283 - accuracy: 0.6859 - auc: 0.8522 - cohen_kappa: 0.5026 - f1_score: 0.6633 - precision: 0.7108 - recall: 0.6170 - val_loss: 0.7126 - val_accuracy: 0.6804 - val_auc: 0.8689 - val_cohen_kappa: 0.4951 - val_f1_score: 0.6556 - val_precision: 0.7605 - val_recall: 0.5959
    Epoch 21/50
    142/142 [==============================] - 41s 290ms/step - loss: 0.7101 - accuracy: 0.6948 - auc: 0.8592 - cohen_kappa: 0.5111 - f1_score: 0.6685 - precision: 0.7240 - recall: 0.6274 - val_loss: 0.7013 - val_accuracy: 0.6825 - val_auc: 0.8667 - val_cohen_kappa: 0.4942 - val_f1_score: 0.6514 - val_precision: 0.7352 - val_recall: 0.6412
    Epoch 22/50
    142/142 [==============================] - 42s 296ms/step - loss: 0.6734 - accuracy: 0.7049 - auc: 0.8703 - cohen_kappa: 0.5336 - f1_score: 0.6840 - precision: 0.7284 - recall: 0.6415 - val_loss: 0.6865 - val_accuracy: 0.7340 - val_auc: 0.8901 - val_cohen_kappa: 0.5974 - val_f1_score: 0.7264 - val_precision: 0.8240 - val_recall: 0.5794
    Epoch 23/50
    142/142 [==============================] - 42s 293ms/step - loss: 0.7146 - accuracy: 0.6751 - auc: 0.8551 - cohen_kappa: 0.4853 - f1_score: 0.6544 - precision: 0.7180 - recall: 0.6177 - val_loss: 0.7501 - val_accuracy: 0.6495 - val_auc: 0.8525 - val_cohen_kappa: 0.4387 - val_f1_score: 0.6128 - val_precision: 0.6868 - val_recall: 0.6330
    Epoch 24/50
    142/142 [==============================] - 44s 307ms/step - loss: 0.6434 - accuracy: 0.7428 - auc: 0.8860 - cohen_kappa: 0.5907 - f1_score: 0.7293 - precision: 0.7629 - recall: 0.6728 - val_loss: 0.6570 - val_accuracy: 0.6990 - val_auc: 0.8848 - val_cohen_kappa: 0.5283 - val_f1_score: 0.6776 - val_precision: 0.7625 - val_recall: 0.6289
    Epoch 25/50
    142/142 [==============================] - 42s 297ms/step - loss: 0.6266 - accuracy: 0.7220 - auc: 0.8887 - cohen_kappa: 0.5562 - f1_score: 0.6950 - precision: 0.7420 - recall: 0.6783 - val_loss: 0.6438 - val_accuracy: 0.7175 - val_auc: 0.8910 - val_cohen_kappa: 0.5518 - val_f1_score: 0.6945 - val_precision: 0.7772 - val_recall: 0.6619
    Epoch 26/50
    142/142 [==============================] - 43s 299ms/step - loss: 0.6257 - accuracy: 0.7377 - auc: 0.8919 - cohen_kappa: 0.5768 - f1_score: 0.7133 - precision: 0.7585 - recall: 0.7015 - val_loss: 0.6539 - val_accuracy: 0.6887 - val_auc: 0.8835 - val_cohen_kappa: 0.5050 - val_f1_score: 0.6604 - val_precision: 0.7434 - val_recall: 0.6392
    Epoch 27/50
    142/142 [==============================] - 42s 294ms/step - loss: 0.5779 - accuracy: 0.7549 - auc: 0.9073 - cohen_kappa: 0.6127 - f1_score: 0.7419 - precision: 0.7747 - recall: 0.7129 - val_loss: 0.5925 - val_accuracy: 0.7443 - val_auc: 0.9114 - val_cohen_kappa: 0.5971 - val_f1_score: 0.7178 - val_precision: 0.8048 - val_recall: 0.6887
    Epoch 28/50
    142/142 [==============================] - 43s 301ms/step - loss: 0.6051 - accuracy: 0.7435 - auc: 0.9004 - cohen_kappa: 0.5900 - f1_score: 0.7222 - precision: 0.7770 - recall: 0.7013 - val_loss: 0.7668 - val_accuracy: 0.6454 - val_auc: 0.8487 - val_cohen_kappa: 0.4282 - val_f1_score: 0.6021 - val_precision: 0.6711 - val_recall: 0.6268
    Epoch 29/50
    142/142 [==============================] - 42s 292ms/step - loss: 0.5882 - accuracy: 0.7566 - auc: 0.9059 - cohen_kappa: 0.6101 - f1_score: 0.7326 - precision: 0.7786 - recall: 0.7136 - val_loss: 0.6402 - val_accuracy: 0.6784 - val_auc: 0.8892 - val_cohen_kappa: 0.4892 - val_f1_score: 0.6511 - val_precision: 0.7056 - val_recall: 0.6474
    Epoch 30/50
    142/142 [==============================] - 42s 296ms/step - loss: 0.5360 - accuracy: 0.7765 - auc: 0.9200 - cohen_kappa: 0.6376 - f1_score: 0.7526 - precision: 0.8008 - recall: 0.7465 - val_loss: 0.5495 - val_accuracy: 0.7753 - val_auc: 0.9183 - val_cohen_kappa: 0.6477 - val_f1_score: 0.7473 - val_precision: 0.8132 - val_recall: 0.7629
    Epoch 31/50
    142/142 [==============================] - 42s 293ms/step - loss: 0.5590 - accuracy: 0.7718 - auc: 0.9152 - cohen_kappa: 0.6360 - f1_score: 0.7498 - precision: 0.8048 - recall: 0.7370 - val_loss: 0.5351 - val_accuracy: 0.8309 - val_auc: 0.9347 - val_cohen_kappa: 0.7380 - val_f1_score: 0.8226 - val_precision: 0.8491 - val_recall: 0.7423
    Epoch 32/50
    142/142 [==============================] - 42s 293ms/step - loss: 0.5532 - accuracy: 0.7520 - auc: 0.9140 - cohen_kappa: 0.6058 - f1_score: 0.7283 - precision: 0.7773 - recall: 0.7107 - val_loss: 0.5180 - val_accuracy: 0.7959 - val_auc: 0.9292 - val_cohen_kappa: 0.6802 - val_f1_score: 0.7791 - val_precision: 0.8206 - val_recall: 0.7546
    Epoch 33/50
    142/142 [==============================] - 43s 306ms/step - loss: 0.5203 - accuracy: 0.7791 - auc: 0.9251 - cohen_kappa: 0.6455 - f1_score: 0.7537 - precision: 0.8022 - recall: 0.7433 - val_loss: 0.4672 - val_accuracy: 0.8082 - val_auc: 0.9429 - val_cohen_kappa: 0.7019 - val_f1_score: 0.7927 - val_precision: 0.8279 - val_recall: 0.7835
    Epoch 34/50
    142/142 [==============================] - 42s 296ms/step - loss: 0.4714 - accuracy: 0.8076 - auc: 0.9382 - cohen_kappa: 0.6959 - f1_score: 0.7903 - precision: 0.8278 - recall: 0.7812 - val_loss: 0.4864 - val_accuracy: 0.8247 - val_auc: 0.9369 - val_cohen_kappa: 0.7261 - val_f1_score: 0.8053 - val_precision: 0.8438 - val_recall: 0.8021
    Epoch 35/50
    142/142 [==============================] - 42s 296ms/step - loss: 0.4961 - accuracy: 0.7879 - auc: 0.9318 - cohen_kappa: 0.6653 - f1_score: 0.7745 - precision: 0.8078 - recall: 0.7614 - val_loss: 0.4444 - val_accuracy: 0.8351 - val_auc: 0.9479 - val_cohen_kappa: 0.7427 - val_f1_score: 0.8200 - val_precision: 0.8491 - val_recall: 0.8124
    Epoch 36/50
    142/142 [==============================] - 42s 293ms/step - loss: 0.4672 - accuracy: 0.8151 - auc: 0.9425 - cohen_kappa: 0.7113 - f1_score: 0.8055 - precision: 0.8391 - recall: 0.7891 - val_loss: 0.3867 - val_accuracy: 0.8495 - val_auc: 0.9593 - val_cohen_kappa: 0.7653 - val_f1_score: 0.8347 - val_precision: 0.8614 - val_recall: 0.8330
    Epoch 37/50
    142/142 [==============================] - 42s 294ms/step - loss: 0.4509 - accuracy: 0.8138 - auc: 0.9438 - cohen_kappa: 0.7087 - f1_score: 0.8020 - precision: 0.8351 - recall: 0.7919 - val_loss: 0.3691 - val_accuracy: 0.8639 - val_auc: 0.9632 - val_cohen_kappa: 0.7892 - val_f1_score: 0.8582 - val_precision: 0.8700 - val_recall: 0.8557
    Epoch 38/50
    142/142 [==============================] - 44s 310ms/step - loss: 0.3757 - accuracy: 0.8550 - auc: 0.9607 - cohen_kappa: 0.7702 - f1_score: 0.8403 - precision: 0.8615 - recall: 0.8391 - val_loss: 0.3513 - val_accuracy: 0.8577 - val_auc: 0.9661 - val_cohen_kappa: 0.7785 - val_f1_score: 0.8490 - val_precision: 0.8675 - val_recall: 0.8371
    Epoch 39/50
    142/142 [==============================] - 42s 294ms/step - loss: 0.3847 - accuracy: 0.8549 - auc: 0.9590 - cohen_kappa: 0.7734 - f1_score: 0.8446 - precision: 0.8700 - recall: 0.8334 - val_loss: 0.3528 - val_accuracy: 0.8742 - val_auc: 0.9652 - val_cohen_kappa: 0.8045 - val_f1_score: 0.8701 - val_precision: 0.8861 - val_recall: 0.8660
    Epoch 40/50
    142/142 [==============================] - 42s 292ms/step - loss: 0.3525 - accuracy: 0.8607 - auc: 0.9654 - cohen_kappa: 0.7796 - f1_score: 0.8495 - precision: 0.8757 - recall: 0.8498 - val_loss: 0.3534 - val_accuracy: 0.8536 - val_auc: 0.9656 - val_cohen_kappa: 0.7739 - val_f1_score: 0.8519 - val_precision: 0.8574 - val_recall: 0.8433
    Epoch 41/50
    142/142 [==============================] - 42s 294ms/step - loss: 0.3334 - accuracy: 0.8645 - auc: 0.9690 - cohen_kappa: 0.7889 - f1_score: 0.8562 - precision: 0.8761 - recall: 0.8531 - val_loss: 0.2648 - val_accuracy: 0.9196 - val_auc: 0.9796 - val_cohen_kappa: 0.8771 - val_f1_score: 0.9173 - val_precision: 0.9223 - val_recall: 0.9052
    Epoch 42/50
    142/142 [==============================] - 42s 296ms/step - loss: 0.3549 - accuracy: 0.8661 - auc: 0.9645 - cohen_kappa: 0.7864 - f1_score: 0.8502 - precision: 0.8743 - recall: 0.8510 - val_loss: 0.2845 - val_accuracy: 0.8928 - val_auc: 0.9781 - val_cohen_kappa: 0.8354 - val_f1_score: 0.8909 - val_precision: 0.8992 - val_recall: 0.8825
    Epoch 43/50
    142/142 [==============================] - 44s 311ms/step - loss: 0.3052 - accuracy: 0.8865 - auc: 0.9754 - cohen_kappa: 0.8205 - f1_score: 0.8783 - precision: 0.8980 - recall: 0.8794 - val_loss: 0.2061 - val_accuracy: 0.9237 - val_auc: 0.9863 - val_cohen_kappa: 0.8826 - val_f1_score: 0.9211 - val_precision: 0.9255 - val_recall: 0.9216
    Epoch 44/50
    142/142 [==============================] - 42s 293ms/step - loss: 0.2597 - accuracy: 0.8974 - auc: 0.9805 - cohen_kappa: 0.8398 - f1_score: 0.8914 - precision: 0.9042 - recall: 0.8932 - val_loss: 0.2215 - val_accuracy: 0.9237 - val_auc: 0.9849 - val_cohen_kappa: 0.8824 - val_f1_score: 0.9212 - val_precision: 0.9274 - val_recall: 0.9216
    Epoch 45/50
    142/142 [==============================] - 42s 295ms/step - loss: 0.2564 - accuracy: 0.9023 - auc: 0.9814 - cohen_kappa: 0.8481 - f1_score: 0.8965 - precision: 0.9081 - recall: 0.8995 - val_loss: 0.2287 - val_accuracy: 0.9175 - val_auc: 0.9842 - val_cohen_kappa: 0.8740 - val_f1_score: 0.9150 - val_precision: 0.9245 - val_recall: 0.9093
    Epoch 46/50
    142/142 [==============================] - 42s 299ms/step - loss: 0.2672 - accuracy: 0.8829 - auc: 0.9802 - cohen_kappa: 0.8158 - f1_score: 0.8753 - precision: 0.8877 - recall: 0.8781 - val_loss: 0.1825 - val_accuracy: 0.9340 - val_auc: 0.9897 - val_cohen_kappa: 0.8988 - val_f1_score: 0.9319 - val_precision: 0.9396 - val_recall: 0.9299
    Epoch 47/50
    142/142 [==============================] - 42s 293ms/step - loss: 0.2879 - accuracy: 0.8871 - auc: 0.9765 - cohen_kappa: 0.8224 - f1_score: 0.8797 - precision: 0.8963 - recall: 0.8825 - val_loss: 0.1673 - val_accuracy: 0.9526 - val_auc: 0.9919 - val_cohen_kappa: 0.9271 - val_f1_score: 0.9505 - val_precision: 0.9523 - val_recall: 0.9464
    Epoch 48/50
    142/142 [==============================] - 41s 290ms/step - loss: 0.2186 - accuracy: 0.9170 - auc: 0.9866 - cohen_kappa: 0.8692 - f1_score: 0.9120 - precision: 0.9224 - recall: 0.9127 - val_loss: 0.2082 - val_accuracy: 0.9299 - val_auc: 0.9861 - val_cohen_kappa: 0.8921 - val_f1_score: 0.9305 - val_precision: 0.9337 - val_recall: 0.9299
    Epoch 49/50
    142/142 [==============================] - 44s 309ms/step - loss: 0.2013 - accuracy: 0.9200 - auc: 0.9883 - cohen_kappa: 0.8733 - f1_score: 0.9154 - precision: 0.9239 - recall: 0.9169 - val_loss: 0.3041 - val_accuracy: 0.8990 - val_auc: 0.9758 - val_cohen_kappa: 0.8431 - val_f1_score: 0.8992 - val_precision: 0.9059 - val_recall: 0.8928
    Epoch 50/50
    142/142 [==============================] - 43s 300ms/step - loss: 0.1896 - accuracy: 0.9258 - auc: 0.9894 - cohen_kappa: 0.8849 - f1_score: 0.9234 - precision: 0.9271 - recall: 0.9241 - val_loss: 0.1395 - val_accuracy: 0.9588 - val_auc: 0.9951 - val_cohen_kappa: 0.9368 - val_f1_score: 0.9570 - val_precision: 0.9588 - val_recall: 0.9588
    


```python
evaluate_model(model, history, test_generator)
```

    
    Test set accuracy: 0.960824728012085 
    
    31/31 [==============================] - 2s 69ms/step
    
                  precision    recall  f1-score   support
    
             AMD       0.91      0.97      0.94       121
             DME       0.98      0.91      0.94       155
          NORMAL       0.98      1.00      0.99       209
    
        accuracy                           0.96       485
       macro avg       0.96      0.96      0.96       485
    weighted avg       0.96      0.96      0.96       485
    
    


![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_31_1.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_31_2.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_31_3.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_31_4.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_31_5.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_31_6.png)


    ROC AUC score: 0.9900738926570206
    


```python
inception_model = generate_model('inceptionv3', 3)

```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
    87916544/87910968 [==============================] - 1s 0us/step
    


```python
inception_model, inception_history = train_model(incres_model, train_generator, val_generator, 20, tf.keras.optimizers.SGD(lr=0.001, momentum=0.9), metrics)
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
    Epoch 1/20
    142/142 [==============================] - 53s 306ms/step - loss: 0.0366 - accuracy: 0.9897 - auc: 0.9963 - cohen_kappa: 0.9677 - f1_score: 0.9774 - precision: 0.9798 - recall: 0.9784 - val_loss: 0.0245 - val_accuracy: 0.9938 - val_auc: 0.9983 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9945 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 2/20
    142/142 [==============================] - 38s 270ms/step - loss: 0.0318 - accuracy: 0.9911 - auc: 0.9985 - cohen_kappa: 0.9862 - f1_score: 0.9909 - precision: 0.9927 - recall: 0.9911 - val_loss: 0.0429 - val_accuracy: 0.9835 - val_auc: 0.9981 - val_cohen_kappa: 0.9747 - val_f1_score: 0.9836 - val_precision: 0.9835 - val_recall: 0.9835
    Epoch 3/20
    142/142 [==============================] - 40s 279ms/step - loss: 0.0140 - accuracy: 0.9945 - auc: 0.9999 - cohen_kappa: 0.9915 - f1_score: 0.9947 - precision: 0.9945 - recall: 0.9945 - val_loss: 0.0175 - val_accuracy: 0.9938 - val_auc: 0.9999 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9939 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 4/20
    142/142 [==============================] - 39s 278ms/step - loss: 0.0123 - accuracy: 0.9961 - auc: 0.9999 - cohen_kappa: 0.9939 - f1_score: 0.9964 - precision: 0.9961 - recall: 0.9961 - val_loss: 0.0259 - val_accuracy: 0.9938 - val_auc: 0.9988 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9939 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 5/20
    142/142 [==============================] - 39s 276ms/step - loss: 0.0326 - accuracy: 0.9891 - auc: 0.9986 - cohen_kappa: 0.9826 - f1_score: 0.9877 - precision: 0.9891 - recall: 0.9891 - val_loss: 0.0047 - val_accuracy: 1.0000 - val_auc: 1.0000 - val_cohen_kappa: 1.0000 - val_f1_score: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
    Epoch 6/20
    142/142 [==============================] - 40s 278ms/step - loss: 0.0190 - accuracy: 0.9909 - auc: 0.9999 - cohen_kappa: 0.9860 - f1_score: 0.9912 - precision: 0.9909 - recall: 0.9909 - val_loss: 0.0062 - val_accuracy: 0.9959 - val_auc: 1.0000 - val_cohen_kappa: 0.9937 - val_f1_score: 0.9964 - val_precision: 0.9959 - val_recall: 0.9959
    Epoch 7/20
    142/142 [==============================] - 41s 283ms/step - loss: 0.0170 - accuracy: 0.9952 - auc: 0.9997 - cohen_kappa: 0.9926 - f1_score: 0.9952 - precision: 0.9954 - recall: 0.9945 - val_loss: 0.0127 - val_accuracy: 0.9959 - val_auc: 1.0000 - val_cohen_kappa: 0.9937 - val_f1_score: 0.9964 - val_precision: 0.9959 - val_recall: 0.9959
    Epoch 8/20
    142/142 [==============================] - 41s 288ms/step - loss: 0.0200 - accuracy: 0.9923 - auc: 0.9998 - cohen_kappa: 0.9881 - f1_score: 0.9920 - precision: 0.9923 - recall: 0.9923 - val_loss: 0.0044 - val_accuracy: 1.0000 - val_auc: 1.0000 - val_cohen_kappa: 1.0000 - val_f1_score: 1.0000 - val_precision: 1.0000 - val_recall: 1.0000
    Epoch 9/20
    142/142 [==============================] - 39s 273ms/step - loss: 0.0199 - accuracy: 0.9935 - auc: 0.9997 - cohen_kappa: 0.9898 - f1_score: 0.9931 - precision: 0.9935 - recall: 0.9935 - val_loss: 0.0065 - val_accuracy: 0.9979 - val_auc: 1.0000 - val_cohen_kappa: 0.9968 - val_f1_score: 0.9982 - val_precision: 0.9979 - val_recall: 0.9979
    Epoch 10/20
    142/142 [==============================] - 39s 277ms/step - loss: 0.0129 - accuracy: 0.9954 - auc: 0.9995 - cohen_kappa: 0.9929 - f1_score: 0.9957 - precision: 0.9954 - recall: 0.9954 - val_loss: 0.0076 - val_accuracy: 0.9938 - val_auc: 1.0000 - val_cohen_kappa: 0.9905 - val_f1_score: 0.9934 - val_precision: 0.9938 - val_recall: 0.9938
    Epoch 11/20
    142/142 [==============================] - 39s 277ms/step - loss: 0.0143 - accuracy: 0.9951 - auc: 0.9996 - cohen_kappa: 0.9923 - f1_score: 0.9954 - precision: 0.9951 - recall: 0.9951 - val_loss: 0.0046 - val_accuracy: 0.9979 - val_auc: 1.0000 - val_cohen_kappa: 0.9968 - val_f1_score: 0.9982 - val_precision: 0.9979 - val_recall: 0.9979
    Epoch 00011: early stopping
    


```python
evaluate_model(inception_model, inception_history, test_generator)
```

    
    Test set accuracy: 0.9979381561279297 
    
    31/31 [==============================] - 6s 83ms/step
    
                  precision    recall  f1-score   support
    
             AMD       1.00      1.00      1.00       121
             DME       0.99      1.00      1.00       155
          NORMAL       1.00      1.00      1.00       209
    
        accuracy                           1.00       485
       macro avg       1.00      1.00      1.00       485
    weighted avg       1.00      1.00      1.00       485
    
    


![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_34_1.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_34_2.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_34_3.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_34_4.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_34_5.png)



![png](2022-06-23-oct-duke-all_files/2022-06-23-oct-duke-all_34_6.png)


    ROC AUC score: 1.0
    
