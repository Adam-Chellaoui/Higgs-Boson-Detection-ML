from implementations import best_parameters_ridge_regression, build_poly, ridge_regression, loadTrainData, loadTestData, mediansByFeature, standardize, cleanData, predictLabels, standardize_submission_dataset

import numpy as np

# We import raw data that we will use to create the model, from the csv file
training_features, training_labels, _ = loadTrainData()

# We calculate the medians of each feature in order to clean the data
medians_model = mediansByFeature(training_features)

# Clean the data
x_model = cleanData(training_features, medians_model)

# Standardization and saving mean and std
x_model, mean_x_model, std_x_model = standardize(x_model)

y_model = training_labels.reshape(training_labels.shape[0],1)

# Removing the outliers (i.e remove the 1/1000 highest and lowest values)
max_thresholds = []
min_thresholds = []
for indx, col in enumerate(x_model.T) :
    max_threshold = np.quantile(col , 0.999)
    max_thresholds.append(max_threshold)
    min_threshold = np.quantile(col, 0.001)
    min_thresholds.append(min_threshold)
    
indices_outliers = set()
for indx_line, line in enumerate(x_model) :
    for indx_col, elem in enumerate(line) :
        if (elem<min_thresholds[indx_col] or elem>max_thresholds[indx_col]) :
            indices_outliers.add(indx_line)

x_model_clean = np.delete(x_model, list(indices_outliers), axis=0)
y_model_clean = np.delete(y_model, list(indices_outliers), axis=0)


#Now we can make a grid search over lambda and the degree, to find the ones that gives the lowest error on test set (using cross validation)
best_degree, best_lambda = best_parameters_ridge_regression(y_model_clean, x_model_clean, k_fold=10, lambdas=np.logspace(-6, -4, 100), degrees=np.arange(1,6))


# Now we determine the weights of our model
x_model_clean_extended = build_poly(x_model_clean, best_degree)
w, loss = ridge_regression(y_model_clean, x_model_clean_extended, best_lambda)


# We import raw data that we will use to finally test the model by submitting the predictions, from the csv file
submission_features, submission_labels, submission_ids = loadTestData()

# Clean the data by replacing meaningless values by the SAME median used while training the model
# Note that during the cleaning, we didn't remove any value, we just replace them (so length of test set remains the same)
x_submission = cleanData(submission_features, medians_model)

# Standardization using the SAME parameters used while training the model
x_submission = standardize_submission_dataset(x_submission, mean_x_model, std_x_model)

# We perfom the polynomial expension of x_submission up to best_degree
x_submission_extended = build_poly(x_submission, best_degree)

#Now that test data has the code shape we can predict labels for the test set and put them inside an csv
submission = np.array(list(predictLabels(x_submission_extended, submission_ids, w)))
np.savetxt("submission.csv", submission, fmt='%d', delimiter=',', header='Id,Prediction', comments='')
