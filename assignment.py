import cv2
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

global basedir, smile, age, glasses, human, hair
# Set the working directory to the base direction
basedir = './'

# Convert function name to the string type
smile = 'smiling'
age = 'young'
glasses = 'eyeglasses'
human = 'human'
hair = 'hair_color'

def rev_noise():
    # In this function, we load images from the dataset and remove any noisy images from the dataset
    # Create two folders, seperate dataset to training set and test set

    # Image path
    images_dir = os.path.join(basedir, 'dataset')

    # Create the face cascade
    face_patterns = cv2.CascadeClassifier(os.path.join(basedir, 'haarcascade_frontalface_default.xml'))

    # Create new folders for training set and test set
    training_set = './training_set/'
    testing_set = './testing_set/'
    if not os.path.exists(training_set):
        os.mkdir(training_set)
    if not os.path.exists(testing_set):
        os.mkdir(testing_set)

    face = []
    count = 1

    # Using for loop to detect each image
    for img in os.listdir(images_dir):
        im = str(count) + '.png'

        # Read the image
        image = cv2.imread(os.path.join(images_dir, im))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_patterns.detectMultiScale(
            gray,
            scaleFactor=1.005,
            minNeighbors=5,
            minSize=(100, 100),
        )

        # split dataset to training set and test set
        if len(faces) != 0:
            if count > 4000:
                cv2.imwrite(os.path.join(testing_set, im), image);
            else:
                cv2.imwrite(os.path.join(training_set, im), image);

            face.append(1)

        else:
            face.append(0)

        count += 1

    # remove label from the dataframe in csv file
    df = pd.read_csv(os.path.join(basedir, 'attribute_list.csv'), header=1)
    df = pd.DataFrame(df)
    df.insert(6, 'face', face)
    df1 = df[-df.face.isin([0])]
    df1.to_csv(os.path.join(basedir, "attribute_list_face.csv"), index=False)

    return 0

def cnn(function):
    # This function build the CNN model and load the training set and test set
    # Split the training set to train and validation
    # Train the model by the train set
    # Evaluate the model by the validation set
    # Tehn, predict the label of test set images
    # Write the results to a csv file

    traindf = pd.read_csv(os.path.join(basedir, 'attribute_list_face.csv'))

    # Fitting the CNN to the images
    datagen = ImageDataGenerator(rescale=1. / 255., validation_split=0.25)

    # Load images to train, validation, and test generator
    train_generator = datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=os.path.join(basedir, 'training_set'),
        x_col='file_name',
        y_col=function,
        has_ext=False,
        subset='training',
        batch_size=5,
        shuffle=False,
        class_mode='categorical',
        target_size=(64, 64))

    valid_generator = datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=os.path.join(basedir, 'training_set'),
        x_col='file_name',
        y_col=function,
        has_ext=False,
        subset='validation',
        batch_size=1,
        shuffle=False,
        class_mode='categorical',
        target_size=(64, 64))

    test_datagen = ImageDataGenerator(rescale=1. / 255.)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=os.path.join(basedir, 'testing_set'),
        x_col='file_name',
        y_col=function,
        has_ext=False,
        batch_size=1,
        shuffle=False,
        class_mode=None,
        target_size=(64, 64))

    # Initialising
    classifier = Sequential()

    # Convolution layers
    classifier.add(Conv2D(32, (3, 3), padding='same',
                          input_shape=(64, 64, 3)))
    classifier.add(Activation('relu'))
    classifier.add(Conv2D(32, (3, 3)))
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))
    classifier.add(Conv2D(64, (3, 3), padding='same'))
    classifier.add(Activation('relu'))
    classifier.add(Conv2D(64, (3, 3)))
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))

    if function == 'hair_color':
        den = 7
    else:
        den = 2

    # Flattening to 1D
    classifier.add(Flatten())

    # Full connection
    classifier.add(Dense(512))
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(den, activation='softmax'))

    # Compiling the CNN
    classifier.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),
                       loss="categorical_crossentropy",
                       metrics=["accuracy"])

    # Fitting the model
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
    clf = classifier.fit_generator(generator=train_generator,
                                   steps_per_epoch=STEP_SIZE_TRAIN,
                                   validation_data=valid_generator,
                                   validation_steps=STEP_SIZE_VALID,
                                   epochs=10
                                   )

    # Evaluate the model
    score = classifier.evaluate_generator(generator=valid_generator, steps=len(valid_generator))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Predict the label of test images
    test_generator.reset()
    pred = classifier.predict_generator(test_generator, steps=len(test_generator), verbose=1)

    predicted_class_indices = np.argmax(pred, axis=1)

    labels = (train_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    if function == smile:
        task = 'task_1.csv'
        fig_acc = 'task_1_acc.png'
        column = traindf.smiling
    elif function == age:
        task = 'task_2.csv'
        fig_acc = 'task_2_acc.png'
        column = traindf.young
    elif function == glasses:
        task = 'task_3.csv'
        fig_acc = 'task_3_acc.png'
        column = traindf.eyeglasses
    elif function == human:
        task = 'task_4.csv'
        fig_acc = 'task_4_acc.png'
        column = traindf.human
    elif function == hair:
        task = 'task_5.csv'
        fig_acc = 'task_5_acc.png'
        column = traindf.hair_color

    # Plotting training curve for accuracy and loss
    plt.plot(clf.history['acc'])
    plt.plot(clf.history['val_acc'])
    plt.plot(clf.history['loss'])
    plt.plot(clf.history['val_loss'])
    plt.title('model accuracy and loss')
    plt.ylabel('acc/loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation', 'train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(basedir, fig_acc))
    plt.close()

    label = list(column)
    del label[0:len(label) - len(predictions)]

    count = 0
    i = 0

    for pre in predictions:
        if label[i] == pre:
            count += 1
        i += 1

    acc = count / len(predictions)

    # Save predicted results to csv file
    # Inference on the Test Set
    filenames = test_generator.filenames
    results = pd.DataFrame({"Filename": filenames,
                            "Predictions": predictions})
    results.columns = ['accuracy',acc]
    results.to_csv(os.path.join(basedir,task), index=False)

    return 0

if __name__ == '__main__':

    # Select function
    print('press number to select the task')
    print('0 for remove noisy images')
    print('1 for Task 1, Emotional recognition')
    print('2 for Task 2, Age identifiction')
    print('3 for Task 3, Glasses detection')
    print('4 for Task 4, Human detection')
    print('5 for Task 5, Hair colour recognition')
    fun_input = input('task to do')
    type(fun_input)

    if fun_input == '0':
        rev_noise()
        print('noisy images have been removed')

    elif fun_input == '1':
        cnn(smile)

    elif fun_input == '2':
        cnn(age)

    elif fun_input == '3':
        cnn(glasses)

    elif fun_input == '4':
        cnn(human)

    elif fun_input == '5':
        cnn(hair)

    print('Task',fun_input,'is done')
