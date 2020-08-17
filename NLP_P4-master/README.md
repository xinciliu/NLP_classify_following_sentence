# NLP_P4
## Part A
For this part, you need to first determine which feature you are going to use. After selected the feature, you need to figure out which dataset you are going to use. 

For development dataset, you need to change the "trainfile" to the path of train dataset and "testfile" to the path of development dataset.  

For the test dataset, you also need to change the "trainfile" to the path of train dataset and "testfile" to the path of test dataset. Besides that, you need to change the code  
"testlenchars1, testlenwords1, testlenchars2, testlenwords2, test_answers, storyid, testending1, testending2 = feature_x(testfile, True)"  
to   
"testlenchars1, testlenwords1, testlenchars2, testlenwords2, _ , storyid, testending1, testending2 = feature_x(testfile, True)"  
, and comment the codes  
"score = classifier.score(test_features, np.array(test_answers))  
 print(score)". 
 
 For both dataset, you may want to change the path of the file you are writing the results to by changing the path in the code 
"with open('partA_Feature1.csv', mode='w') as report_file:".


## Part B
For this part, you need to first convert the training dataset and development dataset by using the file 'transfer_data.py', and convert the test dataset by using the file 'transfer_test_data.py'. Our training, development and testing data should be named 'train.csv, val.csv, test.csv'.

After data conversion, you need to down load the transformers.zip and install transformers in the environment. The installing details are in following link:https://github.com/huggingface/transformers/tree/master/examples#multiple-choice. And since the data is converted to SWAG dataset, we need to run run_multiple_choice.py to training and evaluating the data. We have three parameters: do_train, do_eval, do_test for us to include training, evaluation and testing data.  The other parameters are described in run_multiple_choice.py

Whenever we run do_eval or do_test, the file prediction.csv can be generated to show the prediction result of development dataset or test dataset. The 'prediction.csv' file in this master is the prediction for test.csv which has been provided to us.
