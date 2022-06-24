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
train_path = "../input/oct-dataset-tehran-rabbani-2018"
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

    100%|██████████| 1565/1565 [00:00<00:00, 383528.24it/s]
    100%|██████████| 1585/1585 [00:00<00:00, 378284.50it/s]
    100%|██████████| 1104/1104 [00:00<00:00, 361448.10it/s]



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




![image](https://user-images.githubusercontent.com/37147511/175611378-b95d6616-577e-447b-95b6-5b4898f95612.png)



```python
sns.countplot(x = Y_val)
```




    <AxesSubplot:ylabel='count'>




![image](https://user-images.githubusercontent.com/37147511/175611456-08366bc6-4331-44a5-bd66-eef9be6c59c6.png)



```python
sns.countplot(x = Y_test)
```




    <AxesSubplot:ylabel='count'>




![image](https://user-images.githubusercontent.com/37147511/175611494-c69d8489-a4f4-435d-a9c9-9f55ff92a86b.png)



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

    Found 2977 validated image filenames belonging to 3 classes.
    Found 638 validated image filenames belonging to 3 classes.
    Found 639 validated image filenames belonging to 3 classes.



```python
from tensorflow.keras.applications.vgg16 import VGG16
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D,AveragePooling2D, BatchNormalization, PReLU, ReLU
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications import InceptionResNetV2, ResNet50

def generate_model(pretrained_model = 'vgg16', num_classes=3):
    if pretrained_model == 'inceptionv3':
        base_model = InceptionV3(weights = 'imagenet', include_top=False, input_shape=(224, 224, 3))
    elif pretrained_model == 'inceptionresnet':
        base_model = InceptionResNetV2(weights = 'imagenet', include_top=False, input_shape=(224, 224, 3))
    elif pretrained_model == 'resnet50':
        base_model = ResNet50(weights = 'imagenet', include_top=False, input_shape=(224, 224, 3))        
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
    187/187 [==============================] - 73s 336ms/step - loss: 1.1282 - accuracy: 0.5256 - auc: 0.6997 - cohen_kappa: 0.2614 - f1_score: 0.5022 - precision: 0.5591 - recall: 0.4122 - val_loss: 0.5603 - val_accuracy: 0.7931 - val_auc: 0.9283 - val_cohen_kappa: 0.6869 - val_f1_score: 0.7946 - val_precision: 0.8094 - val_recall: 0.7790
    Epoch 2/50
    187/187 [==============================] - 44s 238ms/step - loss: 0.3665 - accuracy: 0.8644 - auc: 0.9638 - cohen_kappa: 0.7936 - f1_score: 0.8623 - precision: 0.8830 - recall: 0.8454 - val_loss: 0.3506 - val_accuracy: 0.8636 - val_auc: 0.9664 - val_cohen_kappa: 0.7952 - val_f1_score: 0.8607 - val_precision: 0.8688 - val_recall: 0.8511
    Epoch 3/50
    187/187 [==============================] - 45s 242ms/step - loss: 0.3025 - accuracy: 0.8823 - auc: 0.9746 - cohen_kappa: 0.8202 - f1_score: 0.8800 - precision: 0.8919 - recall: 0.8706 - val_loss: 0.3087 - val_accuracy: 0.8887 - val_auc: 0.9768 - val_cohen_kappa: 0.8311 - val_f1_score: 0.8888 - val_precision: 0.8892 - val_recall: 0.8809
    Epoch 4/50
    187/187 [==============================] - 45s 239ms/step - loss: 0.2167 - accuracy: 0.9174 - auc: 0.9868 - cohen_kappa: 0.8736 - f1_score: 0.9156 - precision: 0.9290 - recall: 0.9081 - val_loss: 0.4349 - val_accuracy: 0.8386 - val_auc: 0.9625 - val_cohen_kappa: 0.7529 - val_f1_score: 0.8364 - val_precision: 0.8460 - val_recall: 0.8354
    Epoch 5/50
    187/187 [==============================] - 45s 240ms/step - loss: 0.2020 - accuracy: 0.9248 - auc: 0.9887 - cohen_kappa: 0.8859 - f1_score: 0.9251 - precision: 0.9314 - recall: 0.9200 - val_loss: 0.4493 - val_accuracy: 0.8370 - val_auc: 0.9570 - val_cohen_kappa: 0.7539 - val_f1_score: 0.8370 - val_precision: 0.8424 - val_recall: 0.8292
    Epoch 6/50
    187/187 [==============================] - 46s 244ms/step - loss: 0.1943 - accuracy: 0.9265 - auc: 0.9893 - cohen_kappa: 0.8874 - f1_score: 0.9273 - precision: 0.9360 - recall: 0.9160 - val_loss: 0.1859 - val_accuracy: 0.9185 - val_auc: 0.9904 - val_cohen_kappa: 0.8766 - val_f1_score: 0.9191 - val_precision: 0.9227 - val_recall: 0.9169
    Epoch 7/50
    187/187 [==============================] - 45s 243ms/step - loss: 0.1568 - accuracy: 0.9411 - auc: 0.9926 - cohen_kappa: 0.9100 - f1_score: 0.9406 - precision: 0.9492 - recall: 0.9358 - val_loss: 0.2530 - val_accuracy: 0.9060 - val_auc: 0.9844 - val_cohen_kappa: 0.8572 - val_f1_score: 0.9079 - val_precision: 0.9117 - val_recall: 0.9060
    Epoch 8/50
    187/187 [==============================] - 45s 240ms/step - loss: 0.1521 - accuracy: 0.9491 - auc: 0.9930 - cohen_kappa: 0.9225 - f1_score: 0.9483 - precision: 0.9552 - recall: 0.9445 - val_loss: 0.1425 - val_accuracy: 0.9545 - val_auc: 0.9950 - val_cohen_kappa: 0.9310 - val_f1_score: 0.9537 - val_precision: 0.9618 - val_recall: 0.9483
    Epoch 9/50
    187/187 [==============================] - 46s 246ms/step - loss: 0.1456 - accuracy: 0.9472 - auc: 0.9943 - cohen_kappa: 0.9194 - f1_score: 0.9449 - precision: 0.9552 - recall: 0.9430 - val_loss: 0.1927 - val_accuracy: 0.9357 - val_auc: 0.9899 - val_cohen_kappa: 0.9020 - val_f1_score: 0.9329 - val_precision: 0.9385 - val_recall: 0.9326
    Epoch 10/50
    187/187 [==============================] - 45s 239ms/step - loss: 0.1179 - accuracy: 0.9583 - auc: 0.9960 - cohen_kappa: 0.9363 - f1_score: 0.9577 - precision: 0.9616 - recall: 0.9560 - val_loss: 0.1606 - val_accuracy: 0.9436 - val_auc: 0.9920 - val_cohen_kappa: 0.9147 - val_f1_score: 0.9433 - val_precision: 0.9464 - val_recall: 0.9404
    Epoch 11/50
    187/187 [==============================] - 47s 250ms/step - loss: 0.1182 - accuracy: 0.9573 - auc: 0.9959 - cohen_kappa: 0.9346 - f1_score: 0.9566 - precision: 0.9605 - recall: 0.9540 - val_loss: 0.3605 - val_accuracy: 0.8871 - val_auc: 0.9730 - val_cohen_kappa: 0.8284 - val_f1_score: 0.8894 - val_precision: 0.8866 - val_recall: 0.8824
    Epoch 12/50
    187/187 [==============================] - 45s 240ms/step - loss: 0.0924 - accuracy: 0.9671 - auc: 0.9975 - cohen_kappa: 0.9499 - f1_score: 0.9669 - precision: 0.9685 - recall: 0.9638 - val_loss: 0.2144 - val_accuracy: 0.9310 - val_auc: 0.9861 - val_cohen_kappa: 0.8953 - val_f1_score: 0.9315 - val_precision: 0.9309 - val_recall: 0.9295
    Epoch 13/50
    187/187 [==============================] - 45s 240ms/step - loss: 0.0847 - accuracy: 0.9699 - auc: 0.9971 - cohen_kappa: 0.9544 - f1_score: 0.9696 - precision: 0.9728 - recall: 0.9686 - val_loss: 0.1277 - val_accuracy: 0.9608 - val_auc: 0.9925 - val_cohen_kappa: 0.9405 - val_f1_score: 0.9614 - val_precision: 0.9622 - val_recall: 0.9577
    Epoch 14/50
    187/187 [==============================] - 46s 246ms/step - loss: 0.0961 - accuracy: 0.9636 - auc: 0.9970 - cohen_kappa: 0.9445 - f1_score: 0.9643 - precision: 0.9668 - recall: 0.9614 - val_loss: 0.1645 - val_accuracy: 0.9373 - val_auc: 0.9910 - val_cohen_kappa: 0.9045 - val_f1_score: 0.9366 - val_precision: 0.9446 - val_recall: 0.9357
    Epoch 15/50
    187/187 [==============================] - 45s 240ms/step - loss: 0.0684 - accuracy: 0.9784 - auc: 0.9987 - cohen_kappa: 0.9671 - f1_score: 0.9774 - precision: 0.9819 - recall: 0.9770 - val_loss: 0.1414 - val_accuracy: 0.9498 - val_auc: 0.9946 - val_cohen_kappa: 0.9240 - val_f1_score: 0.9506 - val_precision: 0.9512 - val_recall: 0.9467
    Epoch 16/50
    187/187 [==============================] - 46s 244ms/step - loss: 0.0556 - accuracy: 0.9809 - auc: 0.9992 - cohen_kappa: 0.9707 - f1_score: 0.9817 - precision: 0.9833 - recall: 0.9789 - val_loss: 0.1875 - val_accuracy: 0.9436 - val_auc: 0.9883 - val_cohen_kappa: 0.9144 - val_f1_score: 0.9449 - val_precision: 0.9449 - val_recall: 0.9404
    Epoch 17/50
    187/187 [==============================] - 45s 239ms/step - loss: 0.0718 - accuracy: 0.9736 - auc: 0.9985 - cohen_kappa: 0.9598 - f1_score: 0.9730 - precision: 0.9740 - recall: 0.9710 - val_loss: 0.1400 - val_accuracy: 0.9592 - val_auc: 0.9923 - val_cohen_kappa: 0.9382 - val_f1_score: 0.9603 - val_precision: 0.9592 - val_recall: 0.9577
    Epoch 18/50
    187/187 [==============================] - 45s 243ms/step - loss: 0.0720 - accuracy: 0.9742 - auc: 0.9977 - cohen_kappa: 0.9608 - f1_score: 0.9738 - precision: 0.9767 - recall: 0.9721 - val_loss: 0.1960 - val_accuracy: 0.9357 - val_auc: 0.9894 - val_cohen_kappa: 0.9024 - val_f1_score: 0.9367 - val_precision: 0.9355 - val_recall: 0.9326
    Epoch 19/50
    187/187 [==============================] - 46s 245ms/step - loss: 0.0617 - accuracy: 0.9778 - auc: 0.9988 - cohen_kappa: 0.9662 - f1_score: 0.9775 - precision: 0.9786 - recall: 0.9768 - val_loss: 0.1201 - val_accuracy: 0.9592 - val_auc: 0.9961 - val_cohen_kappa: 0.9382 - val_f1_score: 0.9594 - val_precision: 0.9591 - val_recall: 0.9561
    Epoch 00019: early stopping



```python
evaluate_model(vgg_model, vgg_history, test_generator)
```

    
    Test set accuracy: 0.9655712246894836 
    
    40/40 [==============================] - 3s 77ms/step
    
                  precision    recall  f1-score   support
    
             AMD       1.00      0.92      0.96       254
             DME       0.96      0.99      0.97       173
          NORMAL       0.94      1.00      0.97       212
    
        accuracy                           0.97       639
       macro avg       0.96      0.97      0.97       639
    weighted avg       0.97      0.97      0.97       639
    



![image](https://user-images.githubusercontent.com/37147511/175611586-5a38033c-360f-4d51-9d30-d18647754e56.png)



![image](https://user-images.githubusercontent.com/37147511/175611633-5f35b045-29a8-4af4-8bcf-f75604ca21a0.png)



![image](https://user-images.githubusercontent.com/37147511/175611665-4ca0292e-687b-4a18-a287-48a26ad5839a.png)



![image](https://user-images.githubusercontent.com/37147511/175611692-ff4940e0-7d4c-49db-b084-321d8ad3f50b.png)



![image](https://user-images.githubusercontent.com/37147511/175611717-8e5f5c4b-610b-4705-ad25-2e539b4ebc49.png)


![image](https://user-images.githubusercontent.com/37147511/175611743-52ae0862-ee59-4458-ae67-62261a382495.png)


    ROC AUC score: 0.9990817175287033



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


![image](https://user-images.githubusercontent.com/37147511/175611779-59ca9b40-ef65-41f4-9df3-ec0931599eff.png)



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


![image](https://user-images.githubusercontent.com/37147511/175611851-af6adcd6-bffb-4c10-81dd-d45d4aacfdf9.png)


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


![image](https://user-images.githubusercontent.com/37147511/175612070-bc5e746a-5188-497c-bd0c-eb0f09041a1e.png)



```python
incres_model = generate_model('inceptionresnet', 3)

```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/inception_resnet_v2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5
    219062272/219055592 [==============================] - 1s 0us/step



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
    187/187 [==============================] - 64s 284ms/step - loss: 1.2939 - accuracy: 0.6069 - auc: 0.8986 - cohen_kappa: 0.6223 - f1_score: 0.7492 - precision: 0.7710 - recall: 0.7379 - val_loss: 0.7459 - val_accuracy: 0.7571 - val_auc: 0.9097 - val_cohen_kappa: 0.6262 - val_f1_score: 0.7470 - val_precision: 0.7652 - val_recall: 0.7508
    Epoch 2/50
    187/187 [==============================] - 49s 263ms/step - loss: 0.7114 - accuracy: 0.7767 - auc: 0.9124 - cohen_kappa: 0.6594 - f1_score: 0.7708 - precision: 0.7877 - recall: 0.7571 - val_loss: 0.3950 - val_accuracy: 0.8746 - val_auc: 0.9593 - val_cohen_kappa: 0.8086 - val_f1_score: 0.8706 - val_precision: 0.8837 - val_recall: 0.8574
    Epoch 3/50
    187/187 [==============================] - 48s 257ms/step - loss: 0.3889 - accuracy: 0.8555 - auc: 0.9609 - cohen_kappa: 0.7780 - f1_score: 0.8522 - precision: 0.8727 - recall: 0.8420 - val_loss: 0.3212 - val_accuracy: 0.8840 - val_auc: 0.9717 - val_cohen_kappa: 0.8238 - val_f1_score: 0.8806 - val_precision: 0.8995 - val_recall: 0.8699
    Epoch 4/50
    187/187 [==============================] - 50s 264ms/step - loss: 0.3339 - accuracy: 0.8861 - auc: 0.9705 - cohen_kappa: 0.8265 - f1_score: 0.8841 - precision: 0.8960 - recall: 0.8725 - val_loss: 0.2974 - val_accuracy: 0.9075 - val_auc: 0.9754 - val_cohen_kappa: 0.8591 - val_f1_score: 0.9043 - val_precision: 0.9136 - val_recall: 0.8950
    Epoch 5/50
    187/187 [==============================] - 49s 259ms/step - loss: 0.2744 - accuracy: 0.8979 - auc: 0.9790 - cohen_kappa: 0.8442 - f1_score: 0.8962 - precision: 0.9081 - recall: 0.8887 - val_loss: 0.2680 - val_accuracy: 0.9044 - val_auc: 0.9796 - val_cohen_kappa: 0.8553 - val_f1_score: 0.9010 - val_precision: 0.9095 - val_recall: 0.8981
    Epoch 6/50
    187/187 [==============================] - 49s 263ms/step - loss: 0.2350 - accuracy: 0.9083 - auc: 0.9846 - cohen_kappa: 0.8596 - f1_score: 0.9074 - precision: 0.9169 - recall: 0.8988 - val_loss: 0.2813 - val_accuracy: 0.9060 - val_auc: 0.9776 - val_cohen_kappa: 0.8563 - val_f1_score: 0.8998 - val_precision: 0.9192 - val_recall: 0.8918
    Epoch 7/50
    187/187 [==============================] - 49s 260ms/step - loss: 0.2583 - accuracy: 0.8982 - auc: 0.9822 - cohen_kappa: 0.8451 - f1_score: 0.8972 - precision: 0.9051 - recall: 0.8915 - val_loss: 0.2323 - val_accuracy: 0.9138 - val_auc: 0.9849 - val_cohen_kappa: 0.8692 - val_f1_score: 0.9121 - val_precision: 0.9204 - val_recall: 0.9060
    Epoch 8/50
    187/187 [==============================] - 50s 264ms/step - loss: 0.2245 - accuracy: 0.9157 - auc: 0.9851 - cohen_kappa: 0.8717 - f1_score: 0.9151 - precision: 0.9270 - recall: 0.9086 - val_loss: 0.2206 - val_accuracy: 0.9326 - val_auc: 0.9853 - val_cohen_kappa: 0.8971 - val_f1_score: 0.9288 - val_precision: 0.9366 - val_recall: 0.9263
    Epoch 9/50
    187/187 [==============================] - 49s 263ms/step - loss: 0.1708 - accuracy: 0.9359 - auc: 0.9916 - cohen_kappa: 0.9024 - f1_score: 0.9354 - precision: 0.9445 - recall: 0.9300 - val_loss: 0.2549 - val_accuracy: 0.9028 - val_auc: 0.9823 - val_cohen_kappa: 0.8523 - val_f1_score: 0.9044 - val_precision: 0.9140 - val_recall: 0.8997
    Epoch 10/50
    187/187 [==============================] - 49s 263ms/step - loss: 0.1727 - accuracy: 0.9359 - auc: 0.9914 - cohen_kappa: 0.9026 - f1_score: 0.9357 - precision: 0.9427 - recall: 0.9294 - val_loss: 0.1862 - val_accuracy: 0.9451 - val_auc: 0.9896 - val_cohen_kappa: 0.9163 - val_f1_score: 0.9422 - val_precision: 0.9524 - val_recall: 0.9404
    Epoch 11/50
    187/187 [==============================] - 49s 262ms/step - loss: 0.1435 - accuracy: 0.9446 - auc: 0.9937 - cohen_kappa: 0.9156 - f1_score: 0.9446 - precision: 0.9521 - recall: 0.9404 - val_loss: 0.1508 - val_accuracy: 0.9420 - val_auc: 0.9937 - val_cohen_kappa: 0.9120 - val_f1_score: 0.9411 - val_precision: 0.9464 - val_recall: 0.9404
    Epoch 12/50
    187/187 [==============================] - 50s 269ms/step - loss: 0.1529 - accuracy: 0.9409 - auc: 0.9931 - cohen_kappa: 0.9097 - f1_score: 0.9385 - precision: 0.9473 - recall: 0.9350 - val_loss: 0.1864 - val_accuracy: 0.9326 - val_auc: 0.9901 - val_cohen_kappa: 0.8975 - val_f1_score: 0.9311 - val_precision: 0.9384 - val_recall: 0.9310
    Epoch 13/50
    187/187 [==============================] - 48s 259ms/step - loss: 0.1742 - accuracy: 0.9331 - auc: 0.9914 - cohen_kappa: 0.8982 - f1_score: 0.9325 - precision: 0.9379 - recall: 0.9298 - val_loss: 0.1454 - val_accuracy: 0.9498 - val_auc: 0.9934 - val_cohen_kappa: 0.9239 - val_f1_score: 0.9491 - val_precision: 0.9556 - val_recall: 0.9436
    Epoch 14/50
    187/187 [==============================] - 51s 270ms/step - loss: 0.1269 - accuracy: 0.9518 - auc: 0.9950 - cohen_kappa: 0.9263 - f1_score: 0.9509 - precision: 0.9582 - recall: 0.9458 - val_loss: 0.1666 - val_accuracy: 0.9436 - val_auc: 0.9921 - val_cohen_kappa: 0.9144 - val_f1_score: 0.9420 - val_precision: 0.9447 - val_recall: 0.9373
    Epoch 15/50
    187/187 [==============================] - 48s 258ms/step - loss: 0.1215 - accuracy: 0.9538 - auc: 0.9956 - cohen_kappa: 0.9295 - f1_score: 0.9522 - precision: 0.9575 - recall: 0.9513 - val_loss: 0.1519 - val_accuracy: 0.9451 - val_auc: 0.9928 - val_cohen_kappa: 0.9167 - val_f1_score: 0.9449 - val_precision: 0.9493 - val_recall: 0.9389
    Epoch 16/50
    187/187 [==============================] - 51s 272ms/step - loss: 0.1190 - accuracy: 0.9554 - auc: 0.9950 - cohen_kappa: 0.9319 - f1_score: 0.9559 - precision: 0.9567 - recall: 0.9533 - val_loss: 0.1690 - val_accuracy: 0.9326 - val_auc: 0.9923 - val_cohen_kappa: 0.8976 - val_f1_score: 0.9323 - val_precision: 0.9354 - val_recall: 0.9310
    Epoch 17/50
    187/187 [==============================] - 49s 260ms/step - loss: 0.0960 - accuracy: 0.9596 - auc: 0.9969 - cohen_kappa: 0.9387 - f1_score: 0.9592 - precision: 0.9646 - recall: 0.9577 - val_loss: 0.1486 - val_accuracy: 0.9420 - val_auc: 0.9934 - val_cohen_kappa: 0.9120 - val_f1_score: 0.9397 - val_precision: 0.9525 - val_recall: 0.9420
    Epoch 18/50
    187/187 [==============================] - 51s 272ms/step - loss: 0.1717 - accuracy: 0.9448 - auc: 0.9913 - cohen_kappa: 0.9160 - f1_score: 0.9434 - precision: 0.9469 - recall: 0.9409 - val_loss: 0.1740 - val_accuracy: 0.9404 - val_auc: 0.9909 - val_cohen_kappa: 0.9099 - val_f1_score: 0.9399 - val_precision: 0.9444 - val_recall: 0.9326
    Epoch 19/50
    187/187 [==============================] - 49s 261ms/step - loss: 0.0977 - accuracy: 0.9668 - auc: 0.9969 - cohen_kappa: 0.9495 - f1_score: 0.9663 - precision: 0.9691 - recall: 0.9659 - val_loss: 0.1299 - val_accuracy: 0.9561 - val_auc: 0.9940 - val_cohen_kappa: 0.9335 - val_f1_score: 0.9552 - val_precision: 0.9558 - val_recall: 0.9498
    Epoch 20/50
    187/187 [==============================] - 51s 271ms/step - loss: 0.0911 - accuracy: 0.9697 - auc: 0.9969 - cohen_kappa: 0.9536 - f1_score: 0.9685 - precision: 0.9718 - recall: 0.9688 - val_loss: 0.1648 - val_accuracy: 0.9451 - val_auc: 0.9912 - val_cohen_kappa: 0.9166 - val_f1_score: 0.9441 - val_precision: 0.9479 - val_recall: 0.9420
    Epoch 21/50
    187/187 [==============================] - 49s 260ms/step - loss: 0.1019 - accuracy: 0.9607 - auc: 0.9968 - cohen_kappa: 0.9402 - f1_score: 0.9602 - precision: 0.9630 - recall: 0.9552 - val_loss: 0.1073 - val_accuracy: 0.9608 - val_auc: 0.9956 - val_cohen_kappa: 0.9405 - val_f1_score: 0.9598 - val_precision: 0.9654 - val_recall: 0.9608
    Epoch 22/50
    187/187 [==============================] - 51s 275ms/step - loss: 0.1002 - accuracy: 0.9555 - auc: 0.9969 - cohen_kappa: 0.9322 - f1_score: 0.9540 - precision: 0.9572 - recall: 0.9523 - val_loss: 0.2018 - val_accuracy: 0.9342 - val_auc: 0.9890 - val_cohen_kappa: 0.9005 - val_f1_score: 0.9331 - val_precision: 0.9369 - val_recall: 0.9310
    Epoch 23/50
    187/187 [==============================] - 49s 263ms/step - loss: 0.0897 - accuracy: 0.9704 - auc: 0.9970 - cohen_kappa: 0.9548 - f1_score: 0.9704 - precision: 0.9721 - recall: 0.9682 - val_loss: 0.1721 - val_accuracy: 0.9389 - val_auc: 0.9916 - val_cohen_kappa: 0.9072 - val_f1_score: 0.9367 - val_precision: 0.9432 - val_recall: 0.9373
    Epoch 24/50
    187/187 [==============================] - 51s 273ms/step - loss: 0.1065 - accuracy: 0.9655 - auc: 0.9948 - cohen_kappa: 0.9473 - f1_score: 0.9646 - precision: 0.9680 - recall: 0.9633 - val_loss: 0.1953 - val_accuracy: 0.9483 - val_auc: 0.9875 - val_cohen_kappa: 0.9211 - val_f1_score: 0.9449 - val_precision: 0.9526 - val_recall: 0.9451
    Epoch 25/50
    187/187 [==============================] - 49s 262ms/step - loss: 0.0838 - accuracy: 0.9695 - auc: 0.9978 - cohen_kappa: 0.9536 - f1_score: 0.9686 - precision: 0.9728 - recall: 0.9676 - val_loss: 0.1493 - val_accuracy: 0.9467 - val_auc: 0.9922 - val_cohen_kappa: 0.9192 - val_f1_score: 0.9455 - val_precision: 0.9481 - val_recall: 0.9451
    Epoch 26/50
    187/187 [==============================] - 51s 275ms/step - loss: 0.0836 - accuracy: 0.9709 - auc: 0.9977 - cohen_kappa: 0.9555 - f1_score: 0.9704 - precision: 0.9719 - recall: 0.9699 - val_loss: 0.1426 - val_accuracy: 0.9530 - val_auc: 0.9939 - val_cohen_kappa: 0.9287 - val_f1_score: 0.9512 - val_precision: 0.9573 - val_recall: 0.9483
    Epoch 27/50
    187/187 [==============================] - 49s 263ms/step - loss: 0.0830 - accuracy: 0.9699 - auc: 0.9980 - cohen_kappa: 0.9542 - f1_score: 0.9694 - precision: 0.9717 - recall: 0.9683 - val_loss: 0.1199 - val_accuracy: 0.9577 - val_auc: 0.9959 - val_cohen_kappa: 0.9357 - val_f1_score: 0.9557 - val_precision: 0.9621 - val_recall: 0.9545
    Epoch 00027: early stopping



```python
evaluate_model(incres_model, incresincres_history, test_generator)
```

    
    Test set accuracy: 0.9655712246894836 
    
    40/40 [==============================] - 6s 82ms/step
    
                  precision    recall  f1-score   support
    
             AMD       0.96      0.97      0.97       254
             DME       0.97      0.97      0.97       173
          NORMAL       0.97      0.96      0.96       212
    
        accuracy                           0.97       639
       macro avg       0.97      0.97      0.97       639
    weighted avg       0.97      0.97      0.97       639
    



![image](https://user-images.githubusercontent.com/37147511/175612170-320d52da-d7ed-45e0-919e-3c5a7b669c5e.png)



![image](https://user-images.githubusercontent.com/37147511/175612189-774303d0-bcb5-4894-8403-f6eef120adf7.png)



![image](https://user-images.githubusercontent.com/37147511/175612225-f3ebde00-8294-4e47-8a24-f3200f424215.png)


![image](https://user-images.githubusercontent.com/37147511/175612249-ada0aae7-5414-459b-b70a-2210938d2563.png)



![image](https://user-images.githubusercontent.com/37147511/175612275-8ec47359-c212-49b3-9c47-d273aacfecea.png)


![image](https://user-images.githubusercontent.com/37147511/175612297-7e9ae3f7-8d8c-484b-9037-35f16b70b1d6.png)


    ROC AUC score: 0.996721672251983



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
    187/187 [==============================] - 60s 299ms/step - loss: 1.0918 - accuracy: 0.3753 - auc: 0.7964 - cohen_kappa: 0.3651 - f1_score: 0.5633 - precision: 0.9670 - recall: 0.3551 - val_loss: 1.0881 - val_accuracy: 0.3589 - val_auc: 0.5559 - val_cohen_kappa: 0.0000e+00 - val_f1_score: 0.1761 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/100
    187/187 [==============================] - 55s 296ms/step - loss: 1.0898 - accuracy: 0.3661 - auc: 0.5489 - cohen_kappa: -0.0075 - f1_score: 0.2420 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 1.0885 - val_accuracy: 0.3589 - val_auc: 0.5413 - val_cohen_kappa: 0.0000e+00 - val_f1_score: 0.1761 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 3/100
    187/187 [==============================] - 53s 285ms/step - loss: 1.0792 - accuracy: 0.3983 - auc: 0.5791 - cohen_kappa: 0.0271 - f1_score: 0.2538 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 1.0886 - val_accuracy: 0.3589 - val_auc: 0.5766 - val_cohen_kappa: 0.0000e+00 - val_f1_score: 0.1761 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 4/100
    187/187 [==============================] - 55s 296ms/step - loss: 1.0849 - accuracy: 0.3958 - auc: 0.5701 - cohen_kappa: 0.0455 - f1_score: 0.2609 - precision: 0.0040 - recall: 5.3699e-06 - val_loss: 1.0808 - val_accuracy: 0.4373 - val_auc: 0.5919 - val_cohen_kappa: 0.1198 - val_f1_score: 0.3039 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 5/100
    187/187 [==============================] - 55s 295ms/step - loss: 1.0752 - accuracy: 0.4419 - auc: 0.6002 - cohen_kappa: 0.1031 - f1_score: 0.3263 - precision: 0.4764 - recall: 0.0109 - val_loss: 1.0538 - val_accuracy: 0.4655 - val_auc: 0.6378 - val_cohen_kappa: 0.1597 - val_f1_score: 0.3492 - val_precision: 0.8750 - val_recall: 0.0329
    Epoch 6/100
    187/187 [==============================] - 53s 284ms/step - loss: 1.0519 - accuracy: 0.4583 - auc: 0.6276 - cohen_kappa: 0.1367 - f1_score: 0.3617 - precision: 0.5808 - recall: 0.1271 - val_loss: 1.0564 - val_accuracy: 0.4671 - val_auc: 0.6588 - val_cohen_kappa: 0.1656 - val_f1_score: 0.3671 - val_precision: 1.0000 - val_recall: 0.0157
    Epoch 7/100
    187/187 [==============================] - 56s 297ms/step - loss: 1.0536 - accuracy: 0.4419 - auc: 0.6223 - cohen_kappa: 0.1281 - f1_score: 0.3947 - precision: 0.5831 - recall: 0.1103 - val_loss: 1.0638 - val_accuracy: 0.4781 - val_auc: 0.6443 - val_cohen_kappa: 0.1764 - val_f1_score: 0.3645 - val_precision: 1.0000 - val_recall: 0.0078
    Epoch 8/100
    187/187 [==============================] - 54s 288ms/step - loss: 1.0450 - accuracy: 0.4455 - auc: 0.6332 - cohen_kappa: 0.1184 - f1_score: 0.3674 - precision: 0.5993 - recall: 0.1195 - val_loss: 1.0442 - val_accuracy: 0.4702 - val_auc: 0.6618 - val_cohen_kappa: 0.1691 - val_f1_score: 0.3523 - val_precision: 0.8571 - val_recall: 0.0376
    Epoch 9/100
    187/187 [==============================] - 55s 295ms/step - loss: 1.0454 - accuracy: 0.4488 - auc: 0.6352 - cohen_kappa: 0.1220 - f1_score: 0.3779 - precision: 0.5761 - recall: 0.1129 - val_loss: 1.0198 - val_accuracy: 0.4671 - val_auc: 0.6708 - val_cohen_kappa: 0.1839 - val_f1_score: 0.4228 - val_precision: 0.7073 - val_recall: 0.0909
    Epoch 10/100
    187/187 [==============================] - 54s 287ms/step - loss: 1.0268 - accuracy: 0.4650 - auc: 0.6515 - cohen_kappa: 0.1647 - f1_score: 0.4195 - precision: 0.5801 - recall: 0.1638 - val_loss: 1.0441 - val_accuracy: 0.4655 - val_auc: 0.6618 - val_cohen_kappa: 0.1624 - val_f1_score: 0.3405 - val_precision: 1.0000 - val_recall: 0.0361
    Epoch 11/100
    187/187 [==============================] - 56s 297ms/step - loss: 1.0208 - accuracy: 0.4705 - auc: 0.6577 - cohen_kappa: 0.1710 - f1_score: 0.4173 - precision: 0.5800 - recall: 0.1640 - val_loss: 1.0175 - val_accuracy: 0.5016 - val_auc: 0.6842 - val_cohen_kappa: 0.2133 - val_f1_score: 0.3855 - val_precision: 0.8871 - val_recall: 0.0862
    Epoch 12/100
    187/187 [==============================] - 54s 286ms/step - loss: 1.0073 - accuracy: 0.4685 - auc: 0.6722 - cohen_kappa: 0.1692 - f1_score: 0.4303 - precision: 0.6312 - recall: 0.1971 - val_loss: 0.9911 - val_accuracy: 0.5204 - val_auc: 0.7078 - val_cohen_kappa: 0.2447 - val_f1_score: 0.4276 - val_precision: 0.8913 - val_recall: 0.0643
    Epoch 13/100
    187/187 [==============================] - 56s 299ms/step - loss: 1.0069 - accuracy: 0.4867 - auc: 0.6740 - cohen_kappa: 0.1947 - f1_score: 0.4387 - precision: 0.6099 - recall: 0.1982 - val_loss: 0.9715 - val_accuracy: 0.5188 - val_auc: 0.7149 - val_cohen_kappa: 0.2461 - val_f1_score: 0.4158 - val_precision: 0.6750 - val_recall: 0.1693
    Epoch 14/100
    187/187 [==============================] - 54s 290ms/step - loss: 1.0000 - accuracy: 0.4894 - auc: 0.6828 - cohen_kappa: 0.2050 - f1_score: 0.4575 - precision: 0.5855 - recall: 0.1929 - val_loss: 0.9473 - val_accuracy: 0.5502 - val_auc: 0.7434 - val_cohen_kappa: 0.2998 - val_f1_score: 0.5185 - val_precision: 0.7857 - val_recall: 0.1724
    Epoch 15/100
    187/187 [==============================] - 55s 294ms/step - loss: 0.9494 - accuracy: 0.5356 - auc: 0.7240 - cohen_kappa: 0.2716 - f1_score: 0.5055 - precision: 0.6290 - recall: 0.2898 - val_loss: 1.0658 - val_accuracy: 0.3777 - val_auc: 0.6396 - val_cohen_kappa: 0.0000e+00 - val_f1_score: 0.1828 - val_precision: 0.8889 - val_recall: 0.1003
    Epoch 16/100
    187/187 [==============================] - 54s 289ms/step - loss: 0.9786 - accuracy: 0.5047 - auc: 0.6989 - cohen_kappa: 0.2435 - f1_score: 0.4823 - precision: 0.6880 - recall: 0.2389 - val_loss: 0.9301 - val_accuracy: 0.5705 - val_auc: 0.7639 - val_cohen_kappa: 0.3295 - val_f1_score: 0.5120 - val_precision: 0.8056 - val_recall: 0.1818
    Epoch 17/100
    187/187 [==============================] - 56s 298ms/step - loss: 0.9483 - accuracy: 0.5487 - auc: 0.7306 - cohen_kappa: 0.3002 - f1_score: 0.5311 - precision: 0.6595 - recall: 0.3048 - val_loss: 0.8535 - val_accuracy: 0.6254 - val_auc: 0.7924 - val_cohen_kappa: 0.4285 - val_f1_score: 0.6188 - val_precision: 0.7016 - val_recall: 0.4091
    Epoch 18/100
    187/187 [==============================] - 54s 286ms/step - loss: 0.9085 - accuracy: 0.5793 - auc: 0.7620 - cohen_kappa: 0.3462 - f1_score: 0.5669 - precision: 0.6592 - recall: 0.4031 - val_loss: 0.9743 - val_accuracy: 0.5000 - val_auc: 0.7096 - val_cohen_kappa: 0.2097 - val_f1_score: 0.4224 - val_precision: 0.8088 - val_recall: 0.1724
    Epoch 19/100
    187/187 [==============================] - 56s 299ms/step - loss: 0.8954 - accuracy: 0.5835 - auc: 0.7705 - cohen_kappa: 0.3594 - f1_score: 0.5787 - precision: 0.6815 - recall: 0.3857 - val_loss: 0.9581 - val_accuracy: 0.5486 - val_auc: 0.7342 - val_cohen_kappa: 0.2896 - val_f1_score: 0.4727 - val_precision: 0.6242 - val_recall: 0.4687
    Epoch 20/100
    187/187 [==============================] - 54s 290ms/step - loss: 0.8940 - accuracy: 0.5903 - auc: 0.7703 - cohen_kappa: 0.3681 - f1_score: 0.5901 - precision: 0.6478 - recall: 0.4240 - val_loss: 0.8576 - val_accuracy: 0.6285 - val_auc: 0.7886 - val_cohen_kappa: 0.4255 - val_f1_score: 0.6052 - val_precision: 0.6904 - val_recall: 0.4859
    Epoch 21/100
    187/187 [==============================] - 56s 298ms/step - loss: 0.8578 - accuracy: 0.6048 - auc: 0.7862 - cohen_kappa: 0.3933 - f1_score: 0.6041 - precision: 0.6649 - recall: 0.4611 - val_loss: 1.0005 - val_accuracy: 0.4828 - val_auc: 0.6882 - val_cohen_kappa: 0.1932 - val_f1_score: 0.3963 - val_precision: 0.6267 - val_recall: 0.2210
    Epoch 22/100
    187/187 [==============================] - 54s 290ms/step - loss: 0.8843 - accuracy: 0.5779 - auc: 0.7738 - cohen_kappa: 0.3482 - f1_score: 0.5727 - precision: 0.6561 - recall: 0.4053 - val_loss: 0.8531 - val_accuracy: 0.6238 - val_auc: 0.7953 - val_cohen_kappa: 0.4206 - val_f1_score: 0.6141 - val_precision: 0.6795 - val_recall: 0.4718
    Epoch 23/100
    187/187 [==============================] - 56s 301ms/step - loss: 0.8227 - accuracy: 0.6152 - auc: 0.8047 - cohen_kappa: 0.4138 - f1_score: 0.6185 - precision: 0.6689 - recall: 0.4842 - val_loss: 0.7915 - val_accuracy: 0.6270 - val_auc: 0.8256 - val_cohen_kappa: 0.4288 - val_f1_score: 0.6258 - val_precision: 0.7330 - val_recall: 0.5251
    Epoch 24/100
    187/187 [==============================] - 54s 288ms/step - loss: 0.8318 - accuracy: 0.6131 - auc: 0.8007 - cohen_kappa: 0.4093 - f1_score: 0.6154 - precision: 0.6676 - recall: 0.4803 - val_loss: 0.8390 - val_accuracy: 0.6395 - val_auc: 0.8046 - val_cohen_kappa: 0.4417 - val_f1_score: 0.6268 - val_precision: 0.7162 - val_recall: 0.5063
    Epoch 25/100
    187/187 [==============================] - 57s 302ms/step - loss: 0.7962 - accuracy: 0.6602 - auc: 0.8230 - cohen_kappa: 0.4810 - f1_score: 0.6609 - precision: 0.7049 - recall: 0.5384 - val_loss: 0.7649 - val_accuracy: 0.6661 - val_auc: 0.8361 - val_cohen_kappa: 0.4924 - val_f1_score: 0.6682 - val_precision: 0.6984 - val_recall: 0.5408
    Epoch 26/100
    187/187 [==============================] - 54s 289ms/step - loss: 0.8128 - accuracy: 0.6254 - auc: 0.8132 - cohen_kappa: 0.4281 - f1_score: 0.6266 - precision: 0.6770 - recall: 0.5144 - val_loss: 0.8141 - val_accuracy: 0.6207 - val_auc: 0.8131 - val_cohen_kappa: 0.4104 - val_f1_score: 0.6101 - val_precision: 0.6888 - val_recall: 0.5031
    Epoch 27/100
    187/187 [==============================] - 58s 308ms/step - loss: 0.7756 - accuracy: 0.6370 - auc: 0.8276 - cohen_kappa: 0.4404 - f1_score: 0.6395 - precision: 0.6834 - recall: 0.5515 - val_loss: 0.7183 - val_accuracy: 0.6834 - val_auc: 0.8602 - val_cohen_kappa: 0.5215 - val_f1_score: 0.6819 - val_precision: 0.7510 - val_recall: 0.5909
    Epoch 28/100
    187/187 [==============================] - 54s 289ms/step - loss: 0.7558 - accuracy: 0.6701 - auc: 0.8431 - cohen_kappa: 0.4964 - f1_score: 0.6709 - precision: 0.7210 - recall: 0.5796 - val_loss: 0.7565 - val_accuracy: 0.6489 - val_auc: 0.8368 - val_cohen_kappa: 0.4594 - val_f1_score: 0.6412 - val_precision: 0.7015 - val_recall: 0.5784
    Epoch 29/100
    187/187 [==============================] - 56s 300ms/step - loss: 0.7208 - accuracy: 0.6949 - auc: 0.8586 - cohen_kappa: 0.5366 - f1_score: 0.6967 - precision: 0.7488 - recall: 0.6019 - val_loss: 0.6856 - val_accuracy: 0.7116 - val_auc: 0.8715 - val_cohen_kappa: 0.5582 - val_f1_score: 0.7113 - val_precision: 0.7680 - val_recall: 0.6693
    Epoch 30/100
    187/187 [==============================] - 55s 294ms/step - loss: 0.7108 - accuracy: 0.6873 - auc: 0.8625 - cohen_kappa: 0.5201 - f1_score: 0.6870 - precision: 0.7401 - recall: 0.6210 - val_loss: 0.6669 - val_accuracy: 0.7006 - val_auc: 0.8768 - val_cohen_kappa: 0.5449 - val_f1_score: 0.6980 - val_precision: 0.7295 - val_recall: 0.6426
    Epoch 31/100
    187/187 [==============================] - 57s 304ms/step - loss: 0.7099 - accuracy: 0.6951 - auc: 0.8618 - cohen_kappa: 0.5320 - f1_score: 0.6970 - precision: 0.7397 - recall: 0.6194 - val_loss: 0.6696 - val_accuracy: 0.6928 - val_auc: 0.8774 - val_cohen_kappa: 0.5372 - val_f1_score: 0.6957 - val_precision: 0.7140 - val_recall: 0.6536
    Epoch 32/100
    187/187 [==============================] - 54s 290ms/step - loss: 0.6433 - accuracy: 0.7282 - auc: 0.8880 - cohen_kappa: 0.5863 - f1_score: 0.7329 - precision: 0.7714 - recall: 0.6651 - val_loss: 0.5969 - val_accuracy: 0.7414 - val_auc: 0.9016 - val_cohen_kappa: 0.6129 - val_f1_score: 0.7390 - val_precision: 0.7735 - val_recall: 0.6959
    Epoch 33/100
    187/187 [==============================] - 55s 293ms/step - loss: 0.6447 - accuracy: 0.7217 - auc: 0.8866 - cohen_kappa: 0.5773 - f1_score: 0.7249 - precision: 0.7628 - recall: 0.6675 - val_loss: 0.5540 - val_accuracy: 0.7571 - val_auc: 0.9170 - val_cohen_kappa: 0.6334 - val_f1_score: 0.7596 - val_precision: 0.7878 - val_recall: 0.7100
    Epoch 34/100
    187/187 [==============================] - 55s 293ms/step - loss: 0.5900 - accuracy: 0.7475 - auc: 0.9061 - cohen_kappa: 0.6149 - f1_score: 0.7507 - precision: 0.7776 - recall: 0.7071 - val_loss: 0.6198 - val_accuracy: 0.7163 - val_auc: 0.8948 - val_cohen_kappa: 0.5773 - val_f1_score: 0.7145 - val_precision: 0.7452 - val_recall: 0.6646
    Epoch 35/100
    187/187 [==============================] - 54s 291ms/step - loss: 0.5782 - accuracy: 0.7388 - auc: 0.9085 - cohen_kappa: 0.6027 - f1_score: 0.7436 - precision: 0.7825 - recall: 0.7016 - val_loss: 0.5266 - val_accuracy: 0.7994 - val_auc: 0.9260 - val_cohen_kappa: 0.6946 - val_f1_score: 0.8003 - val_precision: 0.8266 - val_recall: 0.7398
    Epoch 36/100
    187/187 [==============================] - 55s 294ms/step - loss: 0.5517 - accuracy: 0.7607 - auc: 0.9175 - cohen_kappa: 0.6363 - f1_score: 0.7658 - precision: 0.7926 - recall: 0.7141 - val_loss: 0.5032 - val_accuracy: 0.7900 - val_auc: 0.9321 - val_cohen_kappa: 0.6803 - val_f1_score: 0.7928 - val_precision: 0.8117 - val_recall: 0.7633
    Epoch 37/100
    187/187 [==============================] - 52s 279ms/step - loss: 0.5265 - accuracy: 0.7624 - auc: 0.9243 - cohen_kappa: 0.6398 - f1_score: 0.7666 - precision: 0.7998 - recall: 0.7282 - val_loss: 0.4613 - val_accuracy: 0.8025 - val_auc: 0.9417 - val_cohen_kappa: 0.7011 - val_f1_score: 0.8036 - val_precision: 0.8272 - val_recall: 0.7806
    Epoch 38/100
    187/187 [==============================] - 55s 293ms/step - loss: 0.5091 - accuracy: 0.7908 - auc: 0.9298 - cohen_kappa: nan - f1_score: 0.7913 - precision: 0.8208 - recall: 0.7627 - val_loss: 0.4499 - val_accuracy: 0.7931 - val_auc: 0.9449 - val_cohen_kappa: 0.6850 - val_f1_score: 0.7988 - val_precision: 0.8079 - val_recall: 0.7712
    Epoch 39/100
    187/187 [==============================] - 55s 295ms/step - loss: 0.4968 - accuracy: 0.7959 - auc: 0.9329 - cohen_kappa: 0.6905 - f1_score: 0.8016 - precision: 0.8177 - recall: 0.7590 - val_loss: 0.4824 - val_accuracy: 0.7931 - val_auc: 0.9369 - val_cohen_kappa: 0.6870 - val_f1_score: 0.7939 - val_precision: 0.8190 - val_recall: 0.7586
    Epoch 40/100
    187/187 [==============================] - 55s 294ms/step - loss: 0.4528 - accuracy: 0.8143 - auc: 0.9445 - cohen_kappa: 0.7173 - f1_score: 0.8180 - precision: 0.8395 - recall: 0.7861 - val_loss: 0.6806 - val_accuracy: 0.6818 - val_auc: 0.8753 - val_cohen_kappa: 0.5258 - val_f1_score: 0.6755 - val_precision: 0.7135 - val_recall: 0.6442
    Epoch 41/100
    187/187 [==============================] - 55s 293ms/step - loss: 0.5480 - accuracy: 0.7765 - auc: 0.9194 - cohen_kappa: 0.6576 - f1_score: 0.7791 - precision: 0.8132 - recall: 0.7472 - val_loss: 0.3697 - val_accuracy: 0.8401 - val_auc: 0.9635 - val_cohen_kappa: 0.7583 - val_f1_score: 0.8417 - val_precision: 0.8569 - val_recall: 0.8072
    Epoch 42/100
    187/187 [==============================] - 52s 279ms/step - loss: 0.4124 - accuracy: 0.8274 - auc: 0.9536 - cohen_kappa: 0.7370 - f1_score: 0.8309 - precision: 0.8492 - recall: 0.8085 - val_loss: 0.3893 - val_accuracy: 0.8558 - val_auc: 0.9590 - val_cohen_kappa: 0.7820 - val_f1_score: 0.8561 - val_precision: 0.8685 - val_recall: 0.8386
    Epoch 43/100
    187/187 [==============================] - 55s 292ms/step - loss: 0.4171 - accuracy: 0.8293 - auc: 0.9526 - cohen_kappa: 0.7400 - f1_score: 0.8306 - precision: 0.8422 - recall: 0.8058 - val_loss: 0.4821 - val_accuracy: 0.8056 - val_auc: 0.9377 - val_cohen_kappa: 0.7060 - val_f1_score: 0.8036 - val_precision: 0.8279 - val_recall: 0.7915
    Epoch 44/100
    187/187 [==============================] - 54s 291ms/step - loss: 0.4233 - accuracy: 0.8223 - auc: 0.9507 - cohen_kappa: 0.7301 - f1_score: 0.8249 - precision: 0.8448 - recall: 0.8055 - val_loss: 0.3746 - val_accuracy: 0.8401 - val_auc: 0.9608 - val_cohen_kappa: 0.7579 - val_f1_score: 0.8453 - val_precision: 0.8536 - val_recall: 0.8135
    Epoch 45/100
    187/187 [==============================] - 55s 295ms/step - loss: 0.3845 - accuracy: 0.8490 - auc: 0.9600 - cohen_kappa: 0.7706 - f1_score: 0.8500 - precision: 0.8675 - recall: 0.8264 - val_loss: 0.3751 - val_accuracy: 0.8448 - val_auc: 0.9617 - val_cohen_kappa: 0.7636 - val_f1_score: 0.8455 - val_precision: 0.8642 - val_recall: 0.8276
    Epoch 46/100
    187/187 [==============================] - 55s 295ms/step - loss: 0.3787 - accuracy: 0.8510 - auc: 0.9603 - cohen_kappa: 0.7723 - f1_score: 0.8538 - precision: 0.8676 - recall: 0.8337 - val_loss: 0.3189 - val_accuracy: 0.8558 - val_auc: 0.9725 - val_cohen_kappa: 0.7822 - val_f1_score: 0.8595 - val_precision: 0.8725 - val_recall: 0.8370
    Epoch 47/100
    187/187 [==============================] - 51s 274ms/step - loss: 0.3407 - accuracy: 0.8684 - auc: 0.9683 - cohen_kappa: 0.7985 - f1_score: 0.8712 - precision: 0.8811 - recall: 0.8483 - val_loss: 0.3222 - val_accuracy: 0.8715 - val_auc: 0.9718 - val_cohen_kappa: 0.8063 - val_f1_score: 0.8724 - val_precision: 0.8835 - val_recall: 0.8558
    Epoch 48/100
    187/187 [==============================] - 55s 295ms/step - loss: 0.3598 - accuracy: 0.8532 - auc: 0.9647 - cohen_kappa: 0.7761 - f1_score: 0.8539 - precision: 0.8706 - recall: 0.8324 - val_loss: 0.3750 - val_accuracy: 0.8370 - val_auc: 0.9628 - val_cohen_kappa: 0.7537 - val_f1_score: 0.8397 - val_precision: 0.8541 - val_recall: 0.8260
    Epoch 49/100
    187/187 [==============================] - 55s 293ms/step - loss: 0.3451 - accuracy: 0.8617 - auc: 0.9676 - cohen_kappa: 0.7897 - f1_score: 0.8629 - precision: 0.8774 - recall: 0.8435 - val_loss: 0.4346 - val_accuracy: 0.8323 - val_auc: 0.9534 - val_cohen_kappa: 0.7457 - val_f1_score: 0.8322 - val_precision: 0.8457 - val_recall: 0.8245
    Epoch 50/100
    187/187 [==============================] - 55s 293ms/step - loss: 0.3509 - accuracy: 0.8647 - auc: 0.9664 - cohen_kappa: 0.7945 - f1_score: 0.8662 - precision: 0.8770 - recall: 0.8505 - val_loss: 0.2977 - val_accuracy: 0.8699 - val_auc: 0.9757 - val_cohen_kappa: 0.8016 - val_f1_score: 0.8699 - val_precision: 0.8867 - val_recall: 0.8589
    Epoch 51/100
    187/187 [==============================] - 55s 295ms/step - loss: 0.3075 - accuracy: 0.8800 - auc: 0.9741 - cohen_kappa: 0.8170 - f1_score: 0.8824 - precision: 0.8933 - recall: 0.8646 - val_loss: 0.2920 - val_accuracy: 0.8856 - val_auc: 0.9768 - val_cohen_kappa: 0.8258 - val_f1_score: 0.8871 - val_precision: 0.8929 - val_recall: 0.8621
    Epoch 52/100
    187/187 [==============================] - 52s 279ms/step - loss: 0.3287 - accuracy: 0.8708 - auc: 0.9704 - cohen_kappa: 0.8035 - f1_score: 0.8741 - precision: 0.8815 - recall: 0.8544 - val_loss: 0.3612 - val_accuracy: 0.8511 - val_auc: 0.9674 - val_cohen_kappa: 0.7730 - val_f1_score: 0.8507 - val_precision: 0.8628 - val_recall: 0.8480
    Epoch 53/100
    187/187 [==============================] - 55s 292ms/step - loss: 0.2947 - accuracy: 0.8911 - auc: 0.9761 - cohen_kappa: 0.8335 - f1_score: 0.8924 - precision: 0.9056 - recall: 0.8772 - val_loss: 0.2813 - val_accuracy: 0.8824 - val_auc: 0.9787 - val_cohen_kappa: 0.8228 - val_f1_score: 0.8839 - val_precision: 0.9010 - val_recall: 0.8699
    Epoch 54/100
    187/187 [==============================] - 56s 298ms/step - loss: 0.3108 - accuracy: 0.8749 - auc: 0.9731 - cohen_kappa: 0.8103 - f1_score: 0.8763 - precision: 0.8900 - recall: 0.8623 - val_loss: 0.3748 - val_accuracy: 0.8621 - val_auc: 0.9664 - val_cohen_kappa: 0.7894 - val_f1_score: 0.8618 - val_precision: 0.8710 - val_recall: 0.8574
    Epoch 55/100
    187/187 [==============================] - 56s 299ms/step - loss: 0.3443 - accuracy: 0.8710 - auc: 0.9648 - cohen_kappa: 0.8063 - f1_score: 0.8717 - precision: 0.8816 - recall: 0.8553 - val_loss: 0.2433 - val_accuracy: 0.9013 - val_auc: 0.9836 - val_cohen_kappa: 0.8498 - val_f1_score: 0.9035 - val_precision: 0.9084 - val_recall: 0.8856
    Epoch 56/100
    187/187 [==============================] - 55s 295ms/step - loss: 0.2499 - accuracy: 0.9077 - auc: 0.9824 - cohen_kappa: 0.8591 - f1_score: 0.9088 - precision: 0.9165 - recall: 0.8944 - val_loss: 0.2494 - val_accuracy: 0.8997 - val_auc: 0.9833 - val_cohen_kappa: 0.8482 - val_f1_score: 0.9020 - val_precision: 0.9110 - val_recall: 0.8824
    Epoch 57/100
    187/187 [==============================] - 52s 276ms/step - loss: 0.2488 - accuracy: 0.9005 - auc: 0.9831 - cohen_kappa: 0.8482 - f1_score: 0.9015 - precision: 0.9117 - recall: 0.8905 - val_loss: 0.2464 - val_accuracy: 0.8903 - val_auc: 0.9835 - val_cohen_kappa: 0.8342 - val_f1_score: 0.8908 - val_precision: 0.9048 - val_recall: 0.8793
    Epoch 58/100
    187/187 [==============================] - 55s 296ms/step - loss: 0.2614 - accuracy: 0.8924 - auc: 0.9809 - cohen_kappa: 0.8364 - f1_score: 0.8948 - precision: 0.9103 - recall: 0.8846 - val_loss: 0.2266 - val_accuracy: 0.9107 - val_auc: 0.9854 - val_cohen_kappa: 0.8644 - val_f1_score: 0.9118 - val_precision: 0.9187 - val_recall: 0.9028
    Epoch 59/100
    187/187 [==============================] - 55s 294ms/step - loss: 0.2412 - accuracy: 0.9114 - auc: 0.9841 - cohen_kappa: 0.8642 - f1_score: 0.9107 - precision: 0.9185 - recall: 0.9017 - val_loss: 0.2094 - val_accuracy: 0.9138 - val_auc: 0.9876 - val_cohen_kappa: 0.8692 - val_f1_score: 0.9138 - val_precision: 0.9204 - val_recall: 0.9060
    Epoch 60/100
    187/187 [==============================] - 55s 294ms/step - loss: 0.2285 - accuracy: 0.9149 - auc: 0.9850 - cohen_kappa: 0.8699 - f1_score: 0.9161 - precision: 0.9236 - recall: 0.9056 - val_loss: 0.1954 - val_accuracy: 0.9154 - val_auc: 0.9886 - val_cohen_kappa: 0.8717 - val_f1_score: 0.9171 - val_precision: 0.9234 - val_recall: 0.9075
    Epoch 61/100
    187/187 [==============================] - 52s 278ms/step - loss: 0.2199 - accuracy: 0.9198 - auc: 0.9866 - cohen_kappa: 0.8782 - f1_score: 0.9202 - precision: 0.9272 - recall: 0.9113 - val_loss: 0.2468 - val_accuracy: 0.8981 - val_auc: 0.9828 - val_cohen_kappa: 0.8457 - val_f1_score: 0.8996 - val_precision: 0.9053 - val_recall: 0.8840
    Epoch 62/100
    187/187 [==============================] - 55s 293ms/step - loss: 0.2045 - accuracy: 0.9256 - auc: 0.9884 - cohen_kappa: 0.8870 - f1_score: 0.9268 - precision: 0.9328 - recall: 0.9165 - val_loss: 0.2553 - val_accuracy: 0.9028 - val_auc: 0.9820 - val_cohen_kappa: 0.8520 - val_f1_score: 0.9040 - val_precision: 0.9149 - val_recall: 0.8934
    Epoch 63/100
    187/187 [==============================] - 55s 293ms/step - loss: 0.2555 - accuracy: 0.8987 - auc: 0.9817 - cohen_kappa: 0.8451 - f1_score: 0.8982 - precision: 0.9095 - recall: 0.8891 - val_loss: 0.3438 - val_accuracy: 0.8699 - val_auc: 0.9724 - val_cohen_kappa: 0.8009 - val_f1_score: 0.8679 - val_precision: 0.8778 - val_recall: 0.8668
    Epoch 64/100
    187/187 [==============================] - 55s 296ms/step - loss: 0.2160 - accuracy: 0.9177 - auc: 0.9866 - cohen_kappa: 0.8749 - f1_score: 0.9197 - precision: 0.9229 - recall: 0.9115 - val_loss: 0.1656 - val_accuracy: 0.9357 - val_auc: 0.9927 - val_cohen_kappa: 0.9027 - val_f1_score: 0.9365 - val_precision: 0.9455 - val_recall: 0.9248
    Epoch 65/100
    187/187 [==============================] - 55s 296ms/step - loss: 0.2069 - accuracy: 0.9227 - auc: 0.9877 - cohen_kappa: 0.8826 - f1_score: 0.9233 - precision: 0.9285 - recall: 0.9149 - val_loss: 0.1690 - val_accuracy: 0.9295 - val_auc: 0.9920 - val_cohen_kappa: 0.8933 - val_f1_score: 0.9298 - val_precision: 0.9336 - val_recall: 0.9263
    Epoch 66/100
    187/187 [==============================] - 51s 274ms/step - loss: 0.1725 - accuracy: 0.9343 - auc: 0.9917 - cohen_kappa: 0.9001 - f1_score: 0.9349 - precision: 0.9426 - recall: 0.9289 - val_loss: 0.2944 - val_accuracy: 0.8997 - val_auc: 0.9776 - val_cohen_kappa: 0.8478 - val_f1_score: 0.9030 - val_precision: 0.9019 - val_recall: 0.8934
    Epoch 67/100
    187/187 [==============================] - 56s 298ms/step - loss: 0.1889 - accuracy: 0.9284 - auc: 0.9893 - cohen_kappa: 0.8912 - f1_score: 0.9287 - precision: 0.9320 - recall: 0.9203 - val_loss: 0.2485 - val_accuracy: 0.9013 - val_auc: 0.9835 - val_cohen_kappa: 0.8498 - val_f1_score: 0.9030 - val_precision: 0.9006 - val_recall: 0.8950
    Epoch 68/100
    187/187 [==============================] - 55s 294ms/step - loss: 0.2275 - accuracy: 0.9145 - auc: 0.9851 - cohen_kappa: 0.8700 - f1_score: 0.9148 - precision: 0.9242 - recall: 0.9062 - val_loss: 0.1526 - val_accuracy: 0.9451 - val_auc: 0.9925 - val_cohen_kappa: 0.9167 - val_f1_score: 0.9452 - val_precision: 0.9462 - val_recall: 0.9373
    Epoch 69/100
    187/187 [==============================] - 56s 296ms/step - loss: 0.1744 - accuracy: 0.9279 - auc: 0.9911 - cohen_kappa: 0.8897 - f1_score: 0.9274 - precision: 0.9374 - recall: 0.9239 - val_loss: 0.1463 - val_accuracy: 0.9404 - val_auc: 0.9938 - val_cohen_kappa: 0.9097 - val_f1_score: 0.9409 - val_precision: 0.9446 - val_recall: 0.9357
    Epoch 70/100
    187/187 [==============================] - 52s 278ms/step - loss: 0.1730 - accuracy: 0.9281 - auc: 0.9914 - cohen_kappa: 0.8903 - f1_score: 0.9290 - precision: 0.9327 - recall: 0.9233 - val_loss: 0.1393 - val_accuracy: 0.9498 - val_auc: 0.9938 - val_cohen_kappa: 0.9240 - val_f1_score: 0.9504 - val_precision: 0.9542 - val_recall: 0.9467
    Epoch 71/100
    187/187 [==============================] - 55s 296ms/step - loss: 0.1492 - accuracy: 0.9455 - auc: 0.9930 - cohen_kappa: 0.9169 - f1_score: 0.9463 - precision: 0.9476 - recall: 0.9410 - val_loss: 0.1284 - val_accuracy: 0.9498 - val_auc: 0.9946 - val_cohen_kappa: 0.9240 - val_f1_score: 0.9504 - val_precision: 0.9543 - val_recall: 0.9498
    Epoch 72/100
    187/187 [==============================] - 56s 299ms/step - loss: 0.1484 - accuracy: 0.9475 - auc: 0.9936 - cohen_kappa: 0.9202 - f1_score: 0.9489 - precision: 0.9524 - recall: 0.9431 - val_loss: 0.1565 - val_accuracy: 0.9483 - val_auc: 0.9925 - val_cohen_kappa: 0.9214 - val_f1_score: 0.9482 - val_precision: 0.9558 - val_recall: 0.9483
    Epoch 73/100
    187/187 [==============================] - 56s 299ms/step - loss: 0.1500 - accuracy: 0.9483 - auc: 0.9923 - cohen_kappa: 0.9215 - f1_score: 0.9487 - precision: 0.9510 - recall: 0.9410 - val_loss: 0.1570 - val_accuracy: 0.9404 - val_auc: 0.9917 - val_cohen_kappa: 0.9094 - val_f1_score: 0.9419 - val_precision: 0.9433 - val_recall: 0.9389
    Epoch 74/100
    187/187 [==============================] - 55s 293ms/step - loss: 0.1911 - accuracy: 0.9284 - auc: 0.9888 - cohen_kappa: 0.8905 - f1_score: 0.9276 - precision: 0.9340 - recall: 0.9228 - val_loss: 0.1581 - val_accuracy: 0.9451 - val_auc: 0.9923 - val_cohen_kappa: 0.9165 - val_f1_score: 0.9459 - val_precision: 0.9496 - val_recall: 0.9451
    Epoch 75/100
    187/187 [==============================] - 53s 280ms/step - loss: 0.1574 - accuracy: 0.9433 - auc: 0.9919 - cohen_kappa: 0.9135 - f1_score: 0.9438 - precision: 0.9480 - recall: 0.9394 - val_loss: 0.1957 - val_accuracy: 0.9185 - val_auc: 0.9896 - val_cohen_kappa: 0.8763 - val_f1_score: 0.9199 - val_precision: 0.9281 - val_recall: 0.9107
    Epoch 76/100
    187/187 [==============================] - 56s 301ms/step - loss: 0.1455 - accuracy: 0.9536 - auc: 0.9931 - cohen_kappa: 0.9292 - f1_score: 0.9544 - precision: 0.9584 - recall: 0.9456 - val_loss: 0.1551 - val_accuracy: 0.9436 - val_auc: 0.9917 - val_cohen_kappa: 0.9142 - val_f1_score: 0.9448 - val_precision: 0.9436 - val_recall: 0.9436
    Epoch 00076: early stopping



```python
evaluate_model(model, history, test_generator)
```

    
    Test set accuracy: 0.9577465057373047 
    
    40/40 [==============================] - 4s 87ms/step
    
                  precision    recall  f1-score   support
    
             AMD       0.94      0.99      0.97       254
             DME       0.97      0.92      0.95       173
          NORMAL       0.97      0.95      0.96       212
    
        accuracy                           0.96       639
       macro avg       0.96      0.95      0.96       639
    weighted avg       0.96      0.96      0.96       639
    



![image](https://user-images.githubusercontent.com/37147511/175612409-48dd40f7-9268-4a19-aeef-326f86fdff53.png)



![image](https://user-images.githubusercontent.com/37147511/175612440-e7840c99-f0c4-485d-a386-c4a768f947ab.png)



![image](https://user-images.githubusercontent.com/37147511/175612471-df945ed0-1575-4438-be5c-7e53507e5c7a.png)



![image](https://user-images.githubusercontent.com/37147511/175612489-0cf2f4e6-5407-4dcd-b646-21b9063312bd.png)



![image](https://user-images.githubusercontent.com/37147511/175612509-9ddf9f79-a6c5-4cef-b65a-93239e4dd352.png)



![image](https://user-images.githubusercontent.com/37147511/175612532-9006a051-6731-4f0b-9080-ac6642a0f367.png)


    ROC AUC score: 0.9964107654417601



```python
inception_model = generate_model('inceptionv3', 3)

```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
    87916544/87910968 [==============================] - 0s 0us/step



```python
inception_model, inception_history = train_model(inception_model, train_generator, val_generator, 50, tf.keras.optimizers.SGD(lr=0.001, momentum=0.9), metrics)
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
    Epoch 1/50
    187/187 [==============================] - 52s 248ms/step - loss: 3.5068 - accuracy: 0.5491 - auc: 0.8382 - cohen_kappa: 0.5636 - f1_score: 0.7091 - precision: 0.7156 - recall: 0.7089 - val_loss: 0.6122 - val_accuracy: 0.7743 - val_auc: 0.9168 - val_cohen_kappa: 0.6560 - val_f1_score: 0.7743 - val_precision: 0.7967 - val_recall: 0.7555
    Epoch 2/50
    187/187 [==============================] - 49s 262ms/step - loss: 0.7256 - accuracy: 0.7422 - auc: 0.8942 - cohen_kappa: 0.6064 - f1_score: 0.7396 - precision: 0.7589 - recall: 0.7255 - val_loss: 0.5701 - val_accuracy: 0.7915 - val_auc: 0.9248 - val_cohen_kappa: 0.6803 - val_f1_score: 0.7918 - val_precision: 0.7984 - val_recall: 0.7759
    Epoch 3/50
    187/187 [==============================] - 50s 268ms/step - loss: 0.4956 - accuracy: 0.8138 - auc: 0.9395 - cohen_kappa: 0.7165 - f1_score: 0.8109 - precision: 0.8323 - recall: 0.7981 - val_loss: 0.5478 - val_accuracy: 0.8260 - val_auc: 0.9327 - val_cohen_kappa: 0.7358 - val_f1_score: 0.8263 - val_precision: 0.8432 - val_recall: 0.8009
    Epoch 4/50
    187/187 [==============================] - 45s 240ms/step - loss: 0.4546 - accuracy: 0.8183 - auc: 0.9469 - cohen_kappa: 0.7225 - f1_score: 0.8152 - precision: 0.8342 - recall: 0.7968 - val_loss: 0.4533 - val_accuracy: 0.8386 - val_auc: 0.9464 - val_cohen_kappa: 0.7521 - val_f1_score: 0.8330 - val_precision: 0.8652 - val_recall: 0.8150
    Epoch 5/50
    187/187 [==============================] - 49s 264ms/step - loss: 0.4055 - accuracy: 0.8470 - auc: 0.9568 - cohen_kappa: 0.7674 - f1_score: 0.8470 - precision: 0.8696 - recall: 0.8262 - val_loss: 0.4180 - val_accuracy: 0.8339 - val_auc: 0.9542 - val_cohen_kappa: 0.7453 - val_f1_score: 0.8305 - val_precision: 0.8560 - val_recall: 0.8197
    Epoch 6/50
    187/187 [==============================] - 49s 265ms/step - loss: 0.3830 - accuracy: 0.8582 - auc: 0.9607 - cohen_kappa: 0.7842 - f1_score: 0.8567 - precision: 0.8731 - recall: 0.8399 - val_loss: 0.4138 - val_accuracy: 0.8401 - val_auc: 0.9537 - val_cohen_kappa: 0.7550 - val_f1_score: 0.8345 - val_precision: 0.8630 - val_recall: 0.8197
    Epoch 7/50
    187/187 [==============================] - 46s 244ms/step - loss: 0.4147 - accuracy: 0.8424 - auc: 0.9537 - cohen_kappa: 0.7609 - f1_score: 0.8417 - precision: 0.8672 - recall: 0.8157 - val_loss: 0.4279 - val_accuracy: 0.8417 - val_auc: 0.9515 - val_cohen_kappa: 0.7585 - val_f1_score: 0.8406 - val_precision: 0.8597 - val_recall: 0.8260
    Epoch 8/50
    187/187 [==============================] - 50s 267ms/step - loss: 0.4070 - accuracy: 0.8354 - auc: 0.9565 - cohen_kappa: 0.7475 - f1_score: 0.8341 - precision: 0.8597 - recall: 0.8169 - val_loss: 0.3822 - val_accuracy: 0.8589 - val_auc: 0.9608 - val_cohen_kappa: 0.7855 - val_f1_score: 0.8582 - val_precision: 0.8746 - val_recall: 0.8417
    Epoch 9/50
    187/187 [==============================] - 50s 269ms/step - loss: 0.3999 - accuracy: 0.8624 - auc: 0.9581 - cohen_kappa: 0.7887 - f1_score: 0.8576 - precision: 0.8760 - recall: 0.8418 - val_loss: 0.3655 - val_accuracy: 0.8652 - val_auc: 0.9644 - val_cohen_kappa: 0.7958 - val_f1_score: 0.8652 - val_precision: 0.8860 - val_recall: 0.8401
    Epoch 10/50
    187/187 [==============================] - 44s 238ms/step - loss: 0.3252 - accuracy: 0.8745 - auc: 0.9715 - cohen_kappa: 0.8086 - f1_score: 0.8731 - precision: 0.8955 - recall: 0.8527 - val_loss: 0.3900 - val_accuracy: 0.8527 - val_auc: 0.9596 - val_cohen_kappa: 0.7741 - val_f1_score: 0.8488 - val_precision: 0.8725 - val_recall: 0.8260
    Epoch 11/50
    187/187 [==============================] - 50s 269ms/step - loss: 0.3467 - accuracy: 0.8570 - auc: 0.9681 - cohen_kappa: 0.7823 - f1_score: 0.8545 - precision: 0.8833 - recall: 0.8402 - val_loss: 0.4409 - val_accuracy: 0.8339 - val_auc: 0.9539 - val_cohen_kappa: 0.7447 - val_f1_score: 0.8268 - val_precision: 0.8481 - val_recall: 0.8229
    Epoch 12/50
    187/187 [==============================] - 50s 266ms/step - loss: 0.3214 - accuracy: 0.8717 - auc: 0.9721 - cohen_kappa: 0.8036 - f1_score: 0.8693 - precision: 0.8895 - recall: 0.8553 - val_loss: 0.3776 - val_accuracy: 0.8605 - val_auc: 0.9629 - val_cohen_kappa: 0.7868 - val_f1_score: 0.8616 - val_precision: 0.8713 - val_recall: 0.8386
    Epoch 13/50
    187/187 [==============================] - 45s 241ms/step - loss: 0.3288 - accuracy: 0.8653 - auc: 0.9711 - cohen_kappa: 0.7952 - f1_score: 0.8651 - precision: 0.8839 - recall: 0.8482 - val_loss: 0.3709 - val_accuracy: 0.8558 - val_auc: 0.9642 - val_cohen_kappa: 0.7797 - val_f1_score: 0.8574 - val_precision: 0.8709 - val_recall: 0.8354
    Epoch 14/50
    187/187 [==============================] - 50s 268ms/step - loss: 0.3200 - accuracy: 0.8739 - auc: 0.9725 - cohen_kappa: 0.8069 - f1_score: 0.8726 - precision: 0.8892 - recall: 0.8606 - val_loss: 0.3701 - val_accuracy: 0.8558 - val_auc: 0.9647 - val_cohen_kappa: 0.7795 - val_f1_score: 0.8557 - val_precision: 0.8654 - val_recall: 0.8464
    Epoch 15/50
    187/187 [==============================] - 50s 269ms/step - loss: 0.3207 - accuracy: 0.8823 - auc: 0.9724 - cohen_kappa: 0.8210 - f1_score: 0.8819 - precision: 0.8968 - recall: 0.8679 - val_loss: 0.3223 - val_accuracy: 0.8840 - val_auc: 0.9716 - val_cohen_kappa: 0.8232 - val_f1_score: 0.8821 - val_precision: 0.8998 - val_recall: 0.8730
    Epoch 16/50
    187/187 [==============================] - 44s 238ms/step - loss: 0.2964 - accuracy: 0.8831 - auc: 0.9761 - cohen_kappa: 0.8217 - f1_score: 0.8824 - precision: 0.8990 - recall: 0.8606 - val_loss: 0.3605 - val_accuracy: 0.8605 - val_auc: 0.9649 - val_cohen_kappa: 0.7861 - val_f1_score: 0.8592 - val_precision: 0.8766 - val_recall: 0.8354
    Epoch 17/50
    187/187 [==============================] - 50s 268ms/step - loss: 0.2869 - accuracy: 0.8911 - auc: 0.9775 - cohen_kappa: 0.8347 - f1_score: 0.8902 - precision: 0.9048 - recall: 0.8757 - val_loss: 0.3340 - val_accuracy: 0.8809 - val_auc: 0.9705 - val_cohen_kappa: 0.8188 - val_f1_score: 0.8806 - val_precision: 0.8912 - val_recall: 0.8605
    Epoch 18/50
    187/187 [==============================] - 50s 265ms/step - loss: 0.2849 - accuracy: 0.8874 - auc: 0.9778 - cohen_kappa: 0.8280 - f1_score: 0.8859 - precision: 0.9037 - recall: 0.8738 - val_loss: 0.3912 - val_accuracy: 0.8558 - val_auc: 0.9649 - val_cohen_kappa: 0.7785 - val_f1_score: 0.8484 - val_precision: 0.8646 - val_recall: 0.8511
    Epoch 19/50
    187/187 [==============================] - 44s 237ms/step - loss: 0.2891 - accuracy: 0.8890 - auc: 0.9770 - cohen_kappa: 0.8303 - f1_score: 0.8852 - precision: 0.9027 - recall: 0.8779 - val_loss: 0.4598 - val_accuracy: 0.8495 - val_auc: 0.9550 - val_cohen_kappa: 0.7695 - val_f1_score: 0.8525 - val_precision: 0.8546 - val_recall: 0.8386
    Epoch 20/50
    187/187 [==============================] - 50s 265ms/step - loss: 0.2863 - accuracy: 0.8877 - auc: 0.9774 - cohen_kappa: 0.8288 - f1_score: 0.8880 - precision: 0.9010 - recall: 0.8784 - val_loss: 0.3359 - val_accuracy: 0.8683 - val_auc: 0.9694 - val_cohen_kappa: 0.8002 - val_f1_score: 0.8673 - val_precision: 0.8987 - val_recall: 0.8480
    Epoch 21/50
    187/187 [==============================] - 50s 267ms/step - loss: 0.2727 - accuracy: 0.8927 - auc: 0.9797 - cohen_kappa: 0.8367 - f1_score: 0.8930 - precision: 0.9050 - recall: 0.8733 - val_loss: 0.3492 - val_accuracy: 0.8668 - val_auc: 0.9685 - val_cohen_kappa: 0.7958 - val_f1_score: 0.8620 - val_precision: 0.8898 - val_recall: 0.8605
    Epoch 00021: early stopping



```python
evaluate_model(inception_model, inception_history, test_generator)
```

    
    Test set accuracy: 0.8873239159584045 
    
    40/40 [==============================] - 4s 75ms/step
    
                  precision    recall  f1-score   support
    
             AMD       0.87      0.93      0.90       254
             DME       0.96      0.77      0.86       173
          NORMAL       0.87      0.93      0.90       212
    
        accuracy                           0.89       639
       macro avg       0.90      0.88      0.88       639
    weighted avg       0.89      0.89      0.89       639
    



![image](https://user-images.githubusercontent.com/37147511/175612617-916576a3-4479-47cc-bc93-da84fdb6cc86.png)



![image](https://user-images.githubusercontent.com/37147511/175612633-39637dec-18f6-4115-bff6-e5b01ce59835.png)



![image](https://user-images.githubusercontent.com/37147511/175612647-b8fc58a4-bf31-4fbd-82a9-96086ca92a8d.png)



![image](https://user-images.githubusercontent.com/37147511/175612673-1690b8d4-9b7e-4a6f-8853-1cdd7c4d039f.png)



![image](https://user-images.githubusercontent.com/37147511/175612691-3f6698c4-aa19-49ff-8e07-9e9e3375bfe8.png)


![image](https://user-images.githubusercontent.com/37147511/175612722-b31e6180-8df5-4f35-8c8b-f7a95893a99a.png)

    ROC AUC score: 0.983681411328206

