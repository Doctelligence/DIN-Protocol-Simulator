from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import glob
from tensorflow.keras import layers
from tensorflow.keras import Model

import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

from tqdm import tqdm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("\u2022 Using TensorFlow Version:", tf.__version__)
print("\u2022 Using TensorFlow Hub Version: ", hub.__version__)
print('\u2022 GPU Device Found.' if tf.test.is_gpu_available() else '\u2022 GPU Device Not Found. Running on CPU')

tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#tf.config.optimizer.set_jit(True)

#from numpy.random import seed
#seed(16)
#tf.random.set_seed(32)

module_selection = ("mobilenet_v2", 224, 1280)
handle_base, pixels, FV_SIZE = module_selection
MODULE_HANDLE ="https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {} and output dimension {}".format(MODULE_HANDLE, IMAGE_SIZE, FV_SIZE))

#Variables for FL-Training
batch_size = 1
num_epochs = 1
max_rounds = 3
no_patients = 3

Patient_list = '/experiments/medimaging/experimentsabrahamnash/Patients/'
train_dir = []
for i in range(no_patients):
    train_patients = glob.glob(Patient_list + 'Patient_' + str(i))
    train_dir.append(train_patients[0])

print(len(train_dir))
val_dir = '/experiments/datasets/Chest_xray/val'
#test_dir = '/experiments/datasets/Chest_xray/test'
print(train_dir)

# #Centralized
# train_dir = '/experiments/datasets/Chest_xray/train/'
# val_dir = '/experiments/datasets/Chest_xray/val/'

for i in range(no_patients):
  NORMAL_dir = os.path.join(train_dir[i], 'NORMAL')
  NORMAL_fnames = os.listdir(NORMAL_dir)
  print(len(NORMAL_fnames))

def format_example(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255.0

    return x

from keras.preprocessing import image
pose_name = 'NORMAL'
for i in range(no_patients):
  folder = train_dir[i]+'/'+pose_name

images = []
images_org = []
folder = folder
for i in range(no_patients):
  for img in os.listdir(folder):
      img_imp = image.load_img(folder+'/'+img, target_size=(224, 224))
      x = format_example(img_imp)
      images.append(x)
      images_org.append(img_imp)

plt.figure(figsize=(10,8))
plt.imshow(images_org[11], cmap=plt.cm.binary)

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 90,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.1,
                                   zoom_range = 0.1,
                                   brightness_range=(0.4, 1.0),
                                   horizontal_flip = True,
                                   channel_shift_range=80.0,
                                   vertical_flip = True)
# Note that the validation data should not be augmented!
valid_datagen = ImageDataGenerator( rescale = 1.0/255.)

for i in range(no_patients):
  # Flow training images in batches of batch_size using train_datagen generator
  train_generator = train_datagen.flow_from_directory(train_dir[i],
                                                    batch_size = batch_size,
                                                    class_mode = 'binary',
                                                    target_size = (224, 224),
                                                    seed=42,
                                                    shuffle=True)

  # Flow validation images in batches of batch_size using test_datagen generator
validation_generator =  valid_datagen.flow_from_directory(val_dir,
                                                          batch_size  = batch_size,
                                                          class_mode  = 'binary',
                                                          target_size = (224, 224),
                                                          seed=42,
                                                          shuffle=True)

labels_list = list(train_generator.class_indices.keys())
label_dict = train_generator.class_indices

print(labels_list)
label_dict

#Centralized Training
# class myCallback(tf.keras.callbacks.Callback):
#       def on_epoch_end(self, epoch, logs={}):
#         if(logs.get('val_accuracy')>0.999):
#           print("\nReached 99.9% accuracy so cancelling training!")
#           self.model.stop_training = True

Model_array = []
for i in range(no_patients):
  do_fine_tuning = True #'@'param {type:"boolean"}

  feature_extractor = hub.KerasLayer(MODULE_HANDLE, input_shape=IMAGE_SIZE + (3,),
                                   output_shape=[FV_SIZE],
                                   trainable=do_fine_tuning)
  num_classes = 2

  print("Building model with", MODULE_HANDLE)


  model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Dense(num_classes, activation='softmax')])

#  model.summary()
#'@'title (Optional) Unfreeze some layers
  NUM_LAYERS = 8 #'@'param {type:"slider", min:1, max:50, step:1}

  if do_fine_tuning:
      feature_extractor.trainable = True

      for layer in model.layers[-NUM_LAYERS:]:
         layer.trainable = True
  else:
     feature_extractor.trainable = False

  if do_fine_tuning:
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
  else:
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
  Model_array.append(model)
  print(Model_array)

from sklearn.metrics import accuracy_score
from keras.preprocessing import image

for rounds in range(max_rounds):
  for i in range(int(no_patients)):
#Decentralized Training
    Model_array[i].fit(train_generator,epochs=num_epochs, validation_data = validation_generator)
      #  ,callbacks=[callbacks])

  weights = [model.get_weights() for model in Model_array]
  new_weights = []
  wlen = len(weights[0])
  print(wlen)
  for i in range(wlen):
    w_mean = 0
    for j in range(no_patients):
      w_mean = w_mean + weights[j][i]
    new_weights = new_weights + [w_mean / float(no_patients)]

  for model in Model_array:
    model.set_weights(new_weights)
  weights = None

  Global_model = Model_array[no_patients-1]
  print(Global_model)
  from tensorflow.keras.models import load_model
  SAVED_MODEL = "/experiments/medimaging/experimentsabrahamnash/Mobilenetv1-Fl/Global_model/" + str(batch_size) + "_" + str(no_patients) + "_" + str(max_rounds) + "_" + str(num_epochs)

  # Export the SavedModel
  tf.saved_model.save(Global_model, SAVED_MODEL)

  # Convert Using TFLite's Converter
  converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)

  #converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()

  tflite_model_file = str(batch_size) + "_" + str(no_patients) + "_" + str(max_rounds) + "_" + str(num_epochs) + 'Global_model.tflite'

  with open(tflite_model_file, "wb") as f:
    f.write(tflite_model)

  # Load TFLite model and allocate tensors.
  with open(str(batch_size) + "_" + str(no_patients) + "_" + str(max_rounds) + "_" + str(num_epochs) + "Global_model.tflite", 'rb') as fid:
    tflite_model = fid.read()

  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()

  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  #Test Accuracy
  test_folder = '/experiments/datasets/Chest_xray/test'
  os.listdir(test_folder)
  NORMAL_class_name = 'NORMAL'
  PNEUMONIA_class_name = 'PNEUMONIA'
  test_folder_NORMAL = test_folder+'/'+NORMAL_class_name
  test_folder_PNEUMONIA = test_folder+'/'+PNEUMONIA_class_name

  label_dict[NORMAL_class_name]
  label_dict[PNEUMONIA_class_name]
  def get_predictions(folder):
    images = []
    images_org = []
    for img in os.listdir(folder):
        img_imp = image.load_img(folder+'/'+img, target_size=(224, 224))
        x = format_example(img_imp)
        images.append(x)
        images_org.append(img_imp)
    predictions = []
    for i in images:
        interpreter.set_tensor(input_index, i)
        interpreter.invoke()
        prediction_array = interpreter.get_tensor(output_index)
        predictions.append(prediction_array)
    labels= []
    for j in predictions:
        predicted_label = np.argmax(j)
        labels.append(predicted_label)
    return labels

  preds_NORMAL = get_predictions(test_folder_NORMAL)
  preds_PNEUMONIA = get_predictions(test_folder_PNEUMONIA)

  NORMAL_test_dir = os.path.join(test_dir, 'NORMAL')
  NORMAL_test_fnames = os.listdir(NORMAL_test_dir)
  print(len(NORMAL_test_fnames))
    
  PNEUMONIA_test_dir = os.path.join(test_dir, 'PNEUMONIA')
  PNEUMONIA_test_fnames = os.listdir(PNEUMONIA_test_dir)
  print(len(PNEUMONIA_test_fnames))    

  Normal = accuracy_score(len(NORMAL_test_fnames)*[label_dict[NORMAL_class_name]], preds_NORMAL)
  Pneumonia = accuracy_score(len(PNEUMONIA_test_fnames)*[label_dict[PNEUMONIA_class_name]], preds_PNEUMONIA)
  Overall = accuracy_score(len(NORMAL_test_fnames)*[label_dict[NORMAL_class_name]] + 390*[label_dict[PNEUMONIA_class_name]], preds_NORMAL + preds_PNEUMONIA)
  print(Normal)
  print(Pneumonia)
  print(Overall)

  [model.set_weights(new_weights) for model in Model_array]
