## Structure of the project

### This project is divided in 2 files:

- implementations.py contains the implementation of the asked methods, all the methods that are required by them, and methods for model creation
- main.py contains the code that train a model and produce the csv file for submission



## Runing the code

- You need to have numpy installed on your machine to run the code
- To run the code, open a terminal at the location of main.py and enter "python main.py"
- This should generate a file submission.csv in the same location as "main.py"


The data used for the training is from AIcrowd, and should be contained inside the directory data, located at the same location than all other files.
We give you the following data for training inside a zip, you can just unzip it, the location is correct. To summarize, you should have the following hierachy:

* data/
  * -----test.csv
  * -----train.csv
  * -----sample-submission.csv
* implementations.py
* run.py


Note that the runtime of run.py should be in the order of 15 minutes.










