
# Import reqiured pckages and libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, ZeroPadding3D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K
# from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.backend import clear_session
# from tensorflow.compat.v1.keras.backend import get_session

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold

import gc
import tensorflow.python.keras.backend as kb

import matplotlib.image as img
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from IPython.display import Markdown

import datetime
import time

class ModelTrainer():
    
    '''
    A Trainer class which packages methods to train the model and the helper functions and attributes 
    that are necessary for model fit. 
    
    '''
    
    def __init__(self):
        pass
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Purpose: 
        A static method to display markdown formatted output like bold, italic bold etc..
    
    Parameters:
        1. textToDisplay - the string message with formatting styles that is to be displayed
    
    Return Value: 
        NONE
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def PrintMarkdownText(self, textToDisplay):
        display(Markdown('<br>'))
        display(Markdown(textToDisplay))
    
    
    def ClearPreviousKerasSession(self):
        
        '''
        Purpose: 
            Clears the previous Keras sessions. This prevents cluttering of the models. 
            Useful to avoid clutter from old models / layers. Clears the keras training 
            session from the backend & prepares for the next train session. 
            Clearing GPU memory in Keras

        Parameters:
            1. NONE.

        Return Value: 
            NONE
        '''
        
        print("Reset Keras Session Started...")
        sess = kb.get_session()
        kb.clear_session()
        sess.close()
        sess = kb.get_session()
        
        time.sleep(2)
        
        print("Deleting Models from the global space...")
        try:
            del base_model, model # this is from global space - change this as you need
        except:
            pass
        
        time.sleep(2)
        
        print("Clearing Backend Session...")
        
        kb.clear_session()
        
        time.sleep(2)
        
        print("Garbage Collection In Progress...")
        gc.collect()
        
        time.sleep(2)
        print("Reset Keras Session Complete...")
        
        time.sleep(2)
        print(".........................................")
        print("Starting a Fresh Keras Session...")
        # use the same config as you used to create the session
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            
            try:
                for gpu in gpus:
                    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
            
#         config = tf.ConfigProto()
#         config.gpu_options.per_process_gpu_memory_fraction = 1
#         config.gpu_options.visible_device_list = "0"
#         K.set_session(tf.Session(config=config))
        time.sleep(2)
        print("Done...!!!")
        print(".........................................")
        print("New Keras Session Ready to be utilized...")
        
    
    def PrepareImageForPrediction(self, image):
        '''
        Resizes and expands dimension

        Input: image: a 3-channel image as input

        Returns a rank-4 tensor, since the network accepts batches of images
        One image corresponds to batch size of 1
        '''
        img_4d = np.expand_dims(image, axis=0)  # rank 4 tensor for prediction
        return img_4d
        
        
    def GetCallBackList(self, ModelName):
        
        '''
        Purpose: 
            Creates a callback list of the model.

        Parameters:
            1. ModelName - The name of the model that is to be trained.
                           This is needed to create a corresponding Folder to store the model .h5 files.

        Return Value: 
            1. callback_list - List of all callbacks registered for the model.
        '''
        
        filepath = './Models/' + ModelName + '.keras'
        print("Model Checkpoint (.Keras file) Path:", filepath)

        Model_Check_Point = ModelCheckpoint(filepath, 
                                     monitor = 'val_loss', 
                                     verbose = 1, 
                                     save_best_only = True, 
                                     save_weights_only = False, 
                                     mode = 'auto', 
                                     save_freq = 'epoch')

        Learning_Rate = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 2, cooldown = 1, verbose = 1)
        
        Early_Stop = EarlyStopping(monitor = 'val_loss', patience = 7, verbose = 1, mode = 'auto')
        
#         LogsDir = "./TensorBoard/" + ModelName
#         TensorBoardLogs = TensorBoard(log_dir = LogsDir)
#         print("\nTensorBoard Logs Directory:", LogsDir)

        callback_list = [Model_Check_Point, Early_Stop, Learning_Rate]
        return callback_list
    
    def PrintModelCallBacks(self, CallBackList):
        
        '''
        Purpose: 
            Displays a callback list of the model.

        Parameters:
            1. CallBackList - List of all callbacks registered for the model

        Return Value: 
            NONE
        '''
        
        self.PrintMarkdownText("***Model Callback List...***")
        i = 1
        for callBack in CallBackList:
            print(str(i)+'.', callBack)
            i+=1
            
    
    def GetImageDataGenerators(self, train_data, y_train, validation_data, y_validation, BATCH_SIZE):
        
        '''
        Purpose: 
            Create the image generators for training and validation to yield images on the fly during model training.

        Parameters:
            1. train_data - The initial set of model train data.
            2. y_train - The image labels corresponding to the train_data.
            3. validation_data - The initial set of validation data.
            4. y_validation - The image labels corresponding to the validation_data.
            5. BATCH_SIZE - The training batch size.

        Return Value: 
            1. train_gen - Train data generator.
            2. val_gen - Validation data generator.
        '''
        
        train_data_augmentor = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                               rotation_range=90,
                                                                               zoom_range = 0.07,
                                                                               width_shift_range=0.07,
                                                                               height_shift_range=0.07,
                                                                               horizontal_flip=True,
                                                                               vertical_flip=True,
                                                                               fill_mode='nearest')

        validation_data_augmentor = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        train_gen = train_data_augmentor.flow(train_data, y_train, batch_size=BATCH_SIZE, shuffle=True)
        val_gen = validation_data_augmentor.flow(validation_data, y_validation, batch_size=BATCH_SIZE, shuffle=False)

        return train_gen, val_gen
    
    
    def fit(self, model, train_x, train_y, batch_size, num_epochs, val_x, val_y, callback_list):
        
        '''
        Purpose: 
            Trains the model for a fixed number of epochs specified by num_epochs (iterations on a dataset).

        Parameters:
            1. model - The model to be trained.
            2. train_x - The initial set of model train data.
            3. train_y - The image labels corresponding to the train_data.
            4. batch_size - The training batch size.
            5. num_epochs - The number of epochs for which the model will be trained.
            6. val_x - The initial set of validation data.
            7. val_y - The image labels corresponding to the val_x.
            8. callback_list - The call backs registerd for the model.

        Return Value: 
            1. train_history - The model training history object. The train_history.history attribute is a record of 
            training and validation applicable metrics values at successive epochs.
        '''
        
        self.PrintMarkdownText("***Model Training Started...***")

        train_start_time = datetime.datetime.now()
        
        train_history = model.fit(x=train_x, 
                                  y=train_y, 
                                  batch_size=batch_size,
                                  epochs=num_epochs, 
                                  validation_data=(val_x, val_y), 
                                  callbacks=callback_list,
                                  verbose=1)
        
        self.PrintMarkdownText("***Model Training Completed...***")

        train_end_time = datetime.datetime.now()
        training_time = train_end_time - train_start_time
        
        print("Model Training Timespan:", training_time)
        
        # Save the training history to csv file
        train_history_df = pd.DataFrame(train_history.history) 
        train_history_df.index = np.arange(1, len(train_history_df) + 1)
        
        if not os.path.exists('Training History'):
            os.makedirs('Training History')
    
        train_history_csv_file = "./Training History/" + model.name + "_Train_History.csv"
        with open(train_history_csv_file, mode='w') as file:
            train_history_df.to_csv(file)
        
        return train_history
    
    def fit_generator(self, model, train_generator, epochs, validation_generator, callback_list, batch_size=64,
                  total_train_samples=0, total_validation_samples=0):
        '''
        Purpose: 
            Trains the model for a fixed number of epochs specified by num_epochs (iterations on a dataset).
        '''
        
        self.PrintMarkdownText("***Model Training Started...***")

        train_start_time = datetime.datetime.now()
        
        # Check if the generator is the default Keras generator or custom
        if hasattr(train_generator, 'n') and hasattr(train_generator, 'batch_size') and \
        hasattr(validation_generator, 'n') and hasattr(validation_generator, 'batch_size'):
            train_steps_per_epoch = train_generator.n // train_generator.batch_size
            val_steps_per_epoch = validation_generator.n // validation_generator.batch_size
        else:
            train_steps_per_epoch = total_train_samples // batch_size
            val_steps_per_epoch = total_validation_samples // batch_size

        # Train the model
        train_history = model.fit(train_generator,
                                steps_per_epoch=train_steps_per_epoch,
                                epochs=epochs,
                                validation_data=validation_generator,
                                validation_steps=val_steps_per_epoch,
                                callbacks=callback_list,
                                verbose=1)

        train_end_time = datetime.datetime.now()
        training_time = train_end_time - train_start_time

        print("Model Training Timespan:", training_time)
        
        # Check if all arrays in train_history.history are of the same length
        history_dict = train_history.history
        lengths = [len(v) for v in history_dict.values()]
        
        # If lengths are not equal, print a warning and handle the issue
        if len(set(lengths)) != 1:
            print("Warning: Not all arrays in train_history.history have the same length!")
            # Optionally, fill missing entries with NaN or truncate longer arrays to the shortest length
            min_length = min(lengths)
            for key in history_dict:
                history_dict[key] = history_dict[key][:min_length]  # Truncate all to minimum length

        # Convert the history to a DataFrame
        train_history_df = pd.DataFrame(history_dict)
        train_history_df.index = np.arange(1, len(train_history_df) + 1)
        
        # Save the training history to a CSV file
        if not os.path.exists('Training History'):
            os.makedirs('Training History')
        
        train_history_csv_file = f"./Training History/{model.name}_Train_History.csv"
        with open(train_history_csv_file, mode='w') as file:
            train_history_df.to_csv(file)

        return train_history

    
    def plot_model_history(self, train_history):
        
        '''
        Purpose: 
            Plots the model training history.

        Parameters:
            1. train_history - The model training history object. The train_history.history attribute is a record of 
            training and validation applicable metrics values at successive epochs.

        Return Value: 
            NONE
        '''
                           
        self.PrintMarkdownText("***Monitoring Model Train History...***")
                           
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,4))
        axes[0].plot(train_history.history['loss'])   
        axes[0].plot(train_history.history['val_loss'])
        axes[0].legend(['loss','val_loss'])
        axes[0].title.set_text("Model Training/Validation Loss History")

        axes[1].plot(train_history.history['accuracy'])   
        axes[1].plot(train_history.history['val_accuracy'])
        axes[1].legend(['accuracy','val_accuracy'])
        axes[1].title.set_text("Model Training/Validation Accuracy History")
    
    
    def register_swish_activation():
        '''
        Description: 
            Swish: a Self-Gated Activation Function
            Creates & Registers the swish activation
            function in Keras. Updates the Keras 
            custom objects.

            More details @ https://www.bignerdranch.com/blog/implementing-swish-activation-function-in-keras/

            Swish Tech Paper @ https://arxiv.org/abs/1710.05941v1

        Parameters:
            NONE

        Returns:
            NONE
        '''
        from tensorflow.keras.backend import sigmoid
        from keras.utils.generic_utils import get_custom_objects
        from tensorflow.keras.layers import Activation

        def swish(x, beta = 1):
            return (x * sigmoid(beta * x))

        get_custom_objects().update({'swish': Activation(swish)})
        
        

    def PrepareDataForSVMClassification(self, model, intermediate_layer_name, images):

        '''
        Purpose: 
            Prepare the data / image features extracted from the convolutional layer for SVM classification.

        Parameters:
            1. model - The CNN model.
            2. intermediate_layer_name - Convolutional layer name from which the features are to be extracted.
            3. images - The set of image data.

        Return Value: 
            1. X_features_SVM - The convolutional features.
        '''
        
        X_features = []

        # Get the model until the intermmediate_layer (the final convolution layer). 
        # This intermmediate model will act as the feature extractor for final SVM classification. 
        feature_extractor = Model(inputs=model.input, 
                                     outputs=model.get_layer(intermediate_layer_name).output)

        # Extract the feature vector for the image returned by the intermmediate model - the 'feature_extractor'
        # These features will act as the data for the SVM classifier.
        for img in images:
            temp_img = self.PrepareImageForPrediction(img)
            intermediate_output = feature_extractor.predict(temp_img)
            X_features.append(intermediate_output)

        # Prepare the svm training set compatible to the model and the corresponding true labels.
        X_features = np.array(X_features)
        nsamples, dim_x, dim_y = X_features.shape
        X_features_SVM = X_features.reshape((nsamples, dim_x*dim_y))

        return X_features_SVM
    
    
    def Kfold_fit(self, model, X, Y, n_epochs, n_batch_size, n_splits = 5):
        
        '''
        Purpose: 
            K-fold Cross validation for training of the model for each fold.

        Parameters:
            1. model - The CNN model.
            2. X - The train data.
            3. Y - The train image labels.
            4. n_epochs - NUmber of epochs for each fold.
            5. n_batch_size - Training batch size.
            6. n_splits - Number of splits for training data. Default 5.

        Return Value: 
            1. model_train_history - The model training history object. The train_history.history attribute is a record of 
                                     training and validation applicable metrics values at successive epochs 
                                     and successive training folds.
            2. cv_test_acc - The test accuracy of the model for each fold.
            3. cv_test_loss - The test time loss of the model for each fold.
        '''
    
        seed = 8
        np.random.seed(seed)

        cv_test_acc = []
        cv_test_loss = []
        model_train_history = []

        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        CallBackList = self.GetCallBackList(model.name)

        fold_no = 1
        for train, test in kfold.split(X, Y):

            print('----------------------------------------------------------------------------------\n')
            print(f'Training for fold {fold_no} ...')

            train_history = model.fit(X[train],
                                      Y[train],
                                      epochs=n_epochs,
                                      batch_size=n_batch_size,
                                      callbacks=CallBackList,
                                      verbose=1,
                                      validation_split=0.1)

            model_train_history.append(train_history)

            test_scores = model.evaluate(X[test], Y[test], verbose=0)
            
            print('\n----------------------------------------------------------------------------------')
            print(f'Test Time Metrics for Fold {fold_no}\n')
            print("%s: %.2f%%" % (model.metrics_names[1], test_scores[1]*100),
                  "\t%s: %.2f%%" % (model.metrics_names[0], test_scores[0]*100))

            cv_test_acc.append(test_scores[1] * 100)
            cv_test_loss.append(test_scores[0] * 100)

             # Increase fold number
            fold_no = fold_no + 1

        return model_train_history, cv_test_acc, cv_test_loss
             