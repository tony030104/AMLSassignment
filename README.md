# AMLSassignment
> This code is for applied machine learning system module assignment. The aim is to classify images by different features.
> There are 5 tasks in this assignment. Task 1 to Task 4 are binary tasks, Task 5 is a multiclass task.
> There are some noisy images needed to be removed from the dataset, then using convolution neural network to classify images.

## Table of contents
* [Technologies](#technologies)
* [Procedures](#procedures)
* [Contact](#contact)

## Technologies
* Python 3.6
> The code is run under python 3.6 enviroment. There are some libraries used in the code:
* OpenCV
* Keras
* Pandas
* Numpy
* os
* matplotlib
* an XML file, haarcascade_frontalface_default.xml, is required for face detection.

## Procedures
When running the code, the screen will display:

`0 for remove noisy images
 1 for Task 1, Emotional recognition
 2 for Task 2, Age identifiction
 3 for Task 3, Glasses detection
 4 for Task 4, Human detection
 5 for Task 5, Hair colour recognition`
 

Press 0 to remove noisy images from the dataset and seperate them to training set and test set.
This is done by the function rev_noise() in the code.
In the code:

`basedir` is the path where the working directory is. This need to be changed to your working directory.
This folder should contain all document including dataset, attribute_list.csv, haarcascade_frontalface_default.xml
After running rev_noise() function, any images without face are removed.
There are three files created:
* attribute_list_face.csv delete all information without face
* training_set folder contains images for training and validation
* testing_set folder contains images for testing
Then the screen will display:

`noisy images have been removed`

`Task 0 is done`

Run the code again, this time press any number from 1 to 5 to select the task to do.
The directory for test set needs to be changed if you are using other test set.
This can be changed in code directory in test_generator:

`test_generator = test_datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=os.path.join(basedir, 'testing_set'),
        ...)`
        
Or you can change the name of the dataset to testing_set and put the folder into your working directory
You can also change the number of epochs in fit function:

`clf = classifier.fit_generator(...
                                   epochs=10
                                   )`
                                   
When code is done, the screen will display:

`Task X is done`

There are 2 files save in your folder.
* Task_X.csv contains predictions of test images with accuracy at the top
* Task_X_acc.png shows the learning curves graph

## Contact
Created by Yu-Cheng Tsai

email: zceeyts@ucl.ac.uk
