import numpy as np
import os
import glob
# import cv2
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications import VGG19
from keras.regularizers import l1
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

dataset_dir = r'D:\Image_processing'

img_width, img_height = 224, 224

original_files = glob.glob(os.path.join(dataset_dir, "original\*.tif"))
duplicate_files = glob.glob(os.path.join(dataset_dir, "duplicate\*.tif"))

data = []
labels = []
test_data=[]
test_labels=[]
train_original = []
label_original = []

for original_file in original_files:
    img = load_img(original_file, target_size=(img_width, img_height))
    img = img_to_array(img)
    img = preprocess_input(img)
    data.append(img)
    labels.append(0)  # 0 represents original
# data, test_data, labels, test_labels = train_test_split(train_original, label_original, test_size=0.5, random_state=42)

for duplicate_file in duplicate_files:
    img = load_img(duplicate_file, target_size=(img_width, img_height))
    img = img_to_array(img)
    img = preprocess_input(img)
    data.append(img)
    labels.append(1)  # 1 represents duplicate

# for test_file in files_9kV:
#     img = load_img(test_file, target_size=(img_width, img_height))
#     img = img_to_array(img)
#     img = preprocess_input(img)
#     test_data.append(img)
#     test_labels.append(1)

train_data = np.array(data, dtype="float32") / 255.0
test_data = np.array(test_data, dtype="float32") / 255.0
train_labels = np.array(labels)
test_labels = np.array(test_labels)

print(len(train_data), len(test_data))
print(train_data.shape, test_data.shape)

print(f"Unique values in test_labels: {np.unique(test_labels)}")

base_model = VGG19(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))


for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu", kernel_regularizer=l1(0.0001)))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])


initial_perf_vgg_biochip_mixed = model.fit(train_data, train_labels, batch_size=32, validation_data=(test_data,test_labels), epochs=200, verbose=1)

pred_values = model.predict(test_data)
pred_values = np.round(pred_values).flatten()

with open("initial_perf_vgg_biochip_mixed.pkl", "wb") as file:
        pickle.dump(initial_perf_vgg_biochip_mixed.history, file)

model.save("initial_perf_vgg_biochip_mixed.h5")
precision = precision_score(test_labels, pred_values)
recall = recall_score(test_labels, pred_values)
f1 = f1_score(test_labels, pred_values)
fpr, tpr, thresholds = roc_curve(test_labels, pred_values)
roc_auc = auc(fpr, tpr)
validation_metrics = model.evaluate(test_data, test_labels, verbose=1)
print('validation metrics:', validation_metrics)

with open("initial_perf_vgg_biochip_mixed.pkl", "rb") as file:
    model_hist = pickle.load(file)
train_loss = model_hist['loss']
train_accuracy = model_hist['accuracy']
val_loss = model_hist['val_loss']
val_accuracy = model_hist['val_accuracy']
rcParams['figure.figsize'] = (22, 10)
rcParams['axes.spines.top'] = True
rcParams['axes.spines.right'] = True
rcParams.update({'font.size': 20})
rcParams.update({'axes.labelsize': 20})
# plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.plot(train_accuracy, linewidth=5)
plt.plot(val_accuracy, linewidth=5)
plt.title('Model Accuracy', fontsize= 20, weight='bold')
plt.xlabel('Epoch', fontsize= 20, weight='bold')
plt.ylabel('Accuracy', fontsize= 20, weight='bold')
plt.legend(['Train', f'Validation (Acc: {validation_metrics[1]:.2%})'], loc='lower right', handlelength=5, borderpad=2)
plt.subplot(2,2,2)
plt.plot(train_loss, linewidth=5)
plt.plot(val_loss, linewidth=5)
plt.title('Model Loss', fontsize= 20, weight='bold')
plt.xlabel('Epoch', fontsize= 20, weight='bold')
plt.ylabel('Loss', fontsize= 20, weight='bold')
plt.legend(['Train', f'Validation (Loss: {validation_metrics[0]:.2%})'], loc='upper right', handlelength=5, borderpad=2)
plt.subplot(2,2,3)
metrics = ['Precision', 'Recall', 'F1-Score']
values_test= [precision, recall, f1]
bar_width = 0.2
bar_positions1 = np.arange(len(metrics))
bars1 = plt.bar(bar_positions1, values_test, width=bar_width, color=['blue', 'green', 'orange'])
plt.xlabel('Metrics', fontsize= 20, weight='bold')
plt.ylabel('Values', fontsize= 20, weight='bold')
plt.title('Supervised training (original vs. counterfeit instances)', fontsize= 20, weight='bold')
plt.xticks(bar_positions1+ bar_width / 2, metrics)
for bars1, value in zip(bars1, values_test):
    plt.text(bars1.get_x() + bars1.get_width() / 2 - 0.1, bars1.get_height() + 0.02, f'{value:.2f}', ha='center')
plt.subplot(2,2,4)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate', fontsize= 20, weight='bold')
plt.ylabel('True Positive Rate', fontsize= 20, weight='bold')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize= 20, weight='bold')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()