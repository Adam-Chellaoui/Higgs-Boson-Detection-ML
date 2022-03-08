## 1. Description


This project analyzes a dataset and its features using machine learning methods in order to be able to extract meaningful information and to generate predictions using those methods.

The dataset is studied in order to be able to analyze the products that result from the collision of protons at high speeds; the only way to be able to tell if these collions have produced a Higgs boson (whose decay is too fast to let the particle be observed) is by studying the likelihood that a given event’s signature (the products resulted from the decay process) was the result of a Higgs boson (signal) or some other process/particle (background).

A vector will represent the decay signature of a collision event and the models and methods used are going to determine if this event was signal (a Higgs boson) or background (something else) using binary classification.


## 2. Details
This project is divided in 2 files:

- implementations.py contains the implementation of the asked methods, all the methods that are required by them, and methods for model creation
- main.py contains the code that train a model and produce the csv file for submission



## 3. Runing the code

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










