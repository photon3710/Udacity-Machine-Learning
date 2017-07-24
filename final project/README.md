# Tensorflow multi-digit street number recognizer

## The alogorithms and results are saved in 'svhn' folder. 

## Under the 'svhn' folder: 

  Folder 'svhn_model' is for the training of the model on the SVHN dataset. In which, 
    1. svhn.py is for the definition of tenforflow graph, the loss function and train function.
    2. svhn_input.py is for the input of data from the serialization saved file '*.tfrecords' in the svhn_data/svhn_data_recordes_with_extra/
    3. svhn_train.py is the main function for the training of the model. In termianl, run the comman 'python svhn_train.py' under the folder can start the training process.
    4. svhn_eval.py is for the evaluation of the model on the validation or test dataset. One can valuate the predictions of the model by running 'python svhn_eval.py' in the terminal.
    5. svhn_write_graph.py is to export the model into a '*.pb' file which can be easily imported into the android app.

  Folder 'svhn_localizer' is for the training of a localizer on the bounding boxes of the digits.
    1. Start to train the localizer by running 'python svhn_localizer_train.py' in terminal.

  Folder 'svhn_results' is where the cooresponding events and models are saved. 
    1. 'svhn_train' is where the training of the model is saved.

## The data is located in the folder 'svhn_data' on the parent level. Thus the /svhn and /svhn_data should be located under the same folder.


