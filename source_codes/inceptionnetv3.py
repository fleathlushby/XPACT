import numpy as np
import os
import glob
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications import InceptionV3
from keras.regularizers import l1
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

dataset_dir = r'D:\Image_processing\Fingerprinting Work for Biochips\Thickness-wise Labeled Authentic Fingerprints'

img_width, img_height = 224, 224

original_files = glob.glob(os.path.join(dataset_dir, "background_removed_Thickness_5mm\*.jpg"))
duplicate_files = glob.glob(os.path.join(dataset_dir, "background_removed_Thickness_10mm\*.jpg"))
# files_9kV = glob.glob(os.path.join(dataset_dir, "background_removed_9kv\*.jpg"))

# Initialize the data and labels lists
data = []
labels = []
test_data=[]
test_labels=[]
train_original = []
label_original = []

# Load and preprocess the original images
for original_file in original_files:
    img = load_img(original_file, target_size=(img_width, img_height))
    img = img_to_array(img)
    img = preprocess_input(img)
    data.append(img)
    labels.append(0)  # 0 represents original
# data, test_data, labels, test_labels = train_test_split(train_original, label_original, test_size=0.5, random_state=42)

# Load and preprocess the duplicate images
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

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

train_data = np.array(train_data, dtype="float32") / 255.0
test_data = np.array(test_data, dtype="float32") / 255.0
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

print(len(train_data), len(test_data))
print(train_data.shape, test_data.shape)

# Check unique values in test_labels
print(f"Unique values in test_labels: {np.unique(test_labels)}")

base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the base model's layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the base model
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
# model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation="relu", kernel_regularizer=l1(0.001)))
# model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.summary()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

# Train the model
initial_perf_inception_biochip_mixed = model.fit(train_data, train_labels, batch_size=16, validation_data=(test_data,test_labels), epochs=200, verbose=1)

# Check predicted values before rounding
pred_values_raw = model.predict(test_data)
print(f"Predicted values (raw): {pred_values_raw.flatten()[:10]}")

# Check unique values in predicted values after rounding
pred_values = np.round(pred_values_raw).flatten()
print(f"Unique values in pred_values: {np.unique(pred_values)}")

with open("initial_perf_inception_biochip_mixed.pkl", "wb") as file:
        pickle.dump(initial_perf_inception_biochip_mixed.history, file)

model.save("initial_perf_inception_biochip_mixed.h5")
precision = precision_score(test_labels, pred_values)
recall = recall_score(test_labels, pred_values)
f1 = f1_score(test_labels, pred_values)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(test_labels, pred_values_raw)
roc_auc = auc(fpr, tpr)

validation_metrics = model.evaluate(test_data, test_labels, verbose=1)
print('validation metrics:', validation_metrics)

with open("initial_perf_inception_biochip_mixed.pkl", "rb") as file:
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
