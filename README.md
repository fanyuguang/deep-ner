# Named Entity Recognition

* data_utils : create vocabulary, token words labels to ids, prepare train/validation/test datasets
* tfrecords_utils : save data to tensorflow tfrecord file, shuffle read data from tfrecord file
* model : define the inference, loss, accuracy
* train : train the model
* predict : predict the ner result with input sentence, evaluate the result with precision/recall/f score
* evaluate : evaluate the score of model predict, include precision, recall, f

Process:
1. put data into raw-data directory
2. run data_utils.py to create train/validation/test in datasets directory, note that change the value of vocab_size
3. run tfrecords.py, note that change the value of num_steps
4. train the datasets, run "CUDA_VISIBLE_DEVICES='0,1' python train.py", note that change the value of num_classes
5. predict the data, run predict.py, note that change the value of prop_limit to increase precision score, dut this will decrease recall score
6. evaluate the predict result, run evaluate.py
