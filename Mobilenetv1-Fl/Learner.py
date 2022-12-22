import os, glob
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow_hub as hub
from PIL import ImageFile
from keras.preprocessing import image
import numpy as np
from ConfigReader import ConfigReader
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

class Learner:

    def __init__(self,patient_id,no_patients):

        self.model=None

        self.data_root_dir = ""
        self.test_dir = ""
        self.val_dir = ""
        self.model_save_dir = ""

        self.batch_size = 0
        self.num_epochs = 0
        self.max_rounds = 0
        self.no_patients = no_patients

        self.initModel(patient_id, no_patients)

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        print("\u2022 Using TensorFlow Version:", tf.__version__)
        print("\u2022 Using TensorFlow Hub Version: ", hub.__version__)
        print(
            '\u2022 GPU Device Found.' if tf.test.is_gpu_available() else '\u2022 GPU Device Not Found. Running on CPU')

        #tf.config.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        self.module_selection = ("mobilenet_v2", 224, 1280)
        self.handle_base, self.pixels, self.FV_SIZE = self.module_selection
        self.MODULE_HANDLE = "https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(self.handle_base)
        self.IMAGE_SIZE = (self.pixels, self.pixels)
        print("Using {} with input size {} and output dimension {}".format(self.MODULE_HANDLE, self.IMAGE_SIZE, self.FV_SIZE))

        self.fineTuneCreateModel()

    def format_example(self,img):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        return x

    def initModel(self,patient_id,num_patients):
            cr = ConfigReader()
            self.data_root_dir = cr.data_root_dir
            self.test_dir = cr.test_dir
            self.val_dir = cr.val_dir
            self.model_save_dir = cr.model_save_dir

            self.batch_size = cr.batch_size
            self.num_epochs = cr.num_epochs
            self.max_rounds = cr.max_rounds

            self.num_patients = num_patients

            train_patients = glob.glob(self.data_root_dir + '/Patient_' + str(patient_id))
            self.train_dir = train_patients[0]

            data_folder = os.path.join(self.train_dir, 'NORMAL')

            self.images = []
            self.images_org = []

            for img in os.listdir(data_folder):
                img_imp = image.load_img(data_folder + '/' + img, target_size=(224, 224))
                x = self.format_example(img_imp)
                self.images.append(x)
                self.images_org.append(img_imp)

            # Add our data-augmentation parameters to ImageDataGenerator
            train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                               rotation_range=90,
                                               width_shift_range=0.2,
                                               height_shift_range=0.2,
                                               shear_range=0.1,
                                               zoom_range=0.1,
                                               brightness_range=(0.4, 1.0),
                                               horizontal_flip=True,
                                               channel_shift_range=80.0,
                                               vertical_flip=True)

            self.train_generator = train_datagen.flow_from_directory(self.train_dir,
                                                                     batch_size=self.batch_size,
                                                                     class_mode='binary',
                                                                     target_size=(224, 224),
                                                                     seed=42,
                                                                     shuffle=True)

            self.labels_list = list(self.train_generator.class_indices.keys())
            self.label_dict = self.train_generator.class_indices

            # Note that the validation data should not be augmented!
            valid_datagen = ImageDataGenerator(rescale=1.0 / 255.)

            # Flow validation images in batches of batch_size using test_datagen generator
            self.validation_generator = valid_datagen.flow_from_directory(self.val_dir,
                                                                          batch_size=self.batch_size,
                                                                          class_mode='binary',
                                                                          target_size=(224, 224),
                                                                          seed=42,
                                                                          shuffle=True)

    def fineTuneCreateModel(self):
        do_fine_tuning = True  # '@'param {type:"boolean"}

        feature_extractor = hub.KerasLayer(self.MODULE_HANDLE, input_shape=self.IMAGE_SIZE + (3,),
                                           output_shape=[self.FV_SIZE],
                                           trainable=do_fine_tuning)
        num_classes = 2

        print("Building model with", self.MODULE_HANDLE)

        model = tf.keras.Sequential([
            feature_extractor,
            tf.keras.layers.Dense(num_classes, activation='softmax')])

        #  model.summary()
        # '@'title (Optional) Unfreeze some layers
        NUM_LAYERS = 8  # '@'param {type:"slider", min:1, max:50, step:1}

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
        self.model = model

        #create a buffer to store the result of the global aggregation
        self.weights = self.model.get_weights()
        self.model_weight_buffer = np.copy(self.model.get_weights())
        #self.model_weight_buffer = self.model.get_weights()

    def learn(self):
        self.model.fit(self.train_generator, epochs=self.num_epochs, validation_data=self.validation_generator)
        self.weights = self.model.get_weights()
        self.model_weight_buffer = np.copy(self.model.get_weights())

    def setGlobalWeight(self):
        self.model.set_weights(self.model_weight_buffer)

    def saveModel(self,round):
        model_name = self.model_save_dir +"_"+ str(round) + "_" +str(
            self.batch_size) + "_" + str(self.no_patients) + "_" + str(self.max_rounds) + "_" + str(self.num_epochs) + ".h5"
        self.model.save(model_name)
	#tf.saved_model.save(self.model, model_name)
        return model_name

    def get_predictions(self,folder,saved_model_dir):

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()

        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        images = []
        images_org = []
        for img in os.listdir(folder):
            img_imp = image.load_img(folder + '/' + img, target_size=(224, 224))
            x = self.format_example(img_imp)
            images.append(x)
            images_org.append(img_imp)
        predictions = []
        for i in images:
            interpreter.set_tensor(input_index, i)
            interpreter.invoke()
            prediction_array = interpreter.get_tensor(output_index)
            predictions.append(prediction_array)
        labels = []
        for j in predictions:
            predicted_label = np.argmax(j)
            labels.append(predicted_label)
        return labels

    def test(self,round):
        saved_model_dir = self.saveModel(round)
        NORMAL_class_name = 'NORMAL'
        PNEUMONIA_class_name = 'PNEUMONIA'
        test_folder_NORMAL = self.test_dir + '/' + NORMAL_class_name
        test_folder_PNEUMONIA = self.test_dir + '/' + PNEUMONIA_class_name

        preds_NORMAL = self.get_predictions(test_folder_NORMAL,saved_model_dir)
        preds_PNEUMONIA = self.get_predictions(test_folder_PNEUMONIA,saved_model_dir)

        #TODO: What is 234 and 390? Dont HARDCODE. Compute everything.
        NORMAL_test_dir = os.path.join(test_dir, 'NORMAL')
        NORMAL_test_fnames = os.listdir(NORMAL_test_dir)
        print(len(NORMAL_test_fnames))
        
        PNEUMONIA_test_dir = os.path.join(test_dir, 'PNEUMONIA')
        PNEUMONIA_test_fnames = os.listdir(PNEUMONIA_test_dir)
        print(len(PNEUMONIA_test_fnames))
        
        Normal = accuracy_score(len(NORMAL_test_fnames) * [self.label_dict[NORMAL_class_name]], preds_NORMAL)
        Pneumonia = accuracy_score(len(PNEUMONIA_test_fnames) * [self.label_dict[PNEUMONIA_class_name]], preds_PNEUMONIA)
        Overall = accuracy_score(len(NORMAL_test_fnames) * [self.label_dict[NORMAL_class_name]] + len(PNEUMONIA_test_fnames) * [self.label_dict[PNEUMONIA_class_name]],
                                 preds_NORMAL + preds_PNEUMONIA)
        print(Normal)
        print(Pneumonia)
        print(Overall)
