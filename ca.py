import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
from submission_utils import save_history, check_and_prepare_for_submission
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Question 1 functions
def make_positive_data(n_instances, fraction_of_outliers, n_informative_features, n_non_informative_features, std, outliers_std):
    n_inliers = int(n_instances * (1 - fraction_of_outliers))
    n_outliers=int(fraction_of_outliers*n_instances)
    
    # Generate inliers
    informative_inliers = np.random.normal(1, std, (n_inliers, n_informative_features))
    non_informative_inliers = np.random.normal(0, std, (n_inliers, n_non_informative_features))

    # Generate outliers
    informative_outliers = np.random.normal(-1, outliers_std, (n_outliers, n_informative_features))
    non_informative_outliers = np.random.normal(0, std, (n_outliers, n_non_informative_features))

    # Concatenate data
    positive_data = np.concatenate((np.hstack((informative_inliers, non_informative_inliers)),
                                    np.hstack((informative_outliers, non_informative_outliers))))
    return positive_data

def make_negative_data(n_instances, n_informative_features, n_non_informative_features, std):
    informative = np.random.normal(-1, std, (n_instances, n_informative_features))
    non_informative = np.random.normal(0, std, (n_instances, n_non_informative_features))
    negative_data = np.hstack((informative, non_informative))
    return negative_data

def make_dataset(n_instances, fraction_of_outliers, n_informative_features, n_non_informative_features, std, outliers_std):
    half_instances = n_instances // 2
    positive_data = make_positive_data(half_instances, fraction_of_outliers, n_informative_features, n_non_informative_features, std, outliers_std)
    negative_data = make_negative_data(half_instances, n_informative_features, n_non_informative_features, std)
    data_mtx = np.vstack((positive_data, negative_data))
    targets = np.hstack((np.ones(half_instances), np.zeros(half_instances)))
    return data_mtx, targets

def plot2d(data_mtx, targets=None, title='', size=8):
    plt.figure(figsize=(size, size))
   
    if targets is None:
        plt.scatter(data_mtx[:, 0], data_mtx[:, 1])
    else: 
        plt.scatter(data_mtx[:, 0], data_mtx[:, 1], c=targets, cmap='viridis')
    plt.title(title)
    plt.show()  

def plot3d(data_mtx, targets=None, title='', size=8):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection='3d')
    if targets is None:
        ax.scatter(data_mtx[:, 0], data_mtx[:, 1], data_mtx[:, 2])
    else:
        ax.scatter(data_mtx[:, 0], data_mtx[:, 1], data_mtx[:, 2], c=targets, cmap='viridis')
    plt.title(title)
    ax.set_xlabel('Column 1')
    ax.set_ylabel('Column 2')
    ax.set_zlabel('Column 3')
    plt.show()


# Just run the following code, do not modify it
n_instances = 1000
fraction_of_outliers = 0.3
n_informative_features = 1
n_non_informative_features = 2
std = 0.5
outliers_std = .5  # 5 

data_mtx, targets = make_dataset(n_instances, fraction_of_outliers, n_informative_features, n_non_informative_features, std, 2)

# plot2d(data_mtx, targets, title='Data 2D', size=8)
# plot3d(data_mtx, 

# Question 2 functions
def rebalance(X, y):
    unique_values, class_count = np.unique(y, return_counts=True)
    # print(f'Number of distinct classes: {len(class_count)}, {unique_values}')
    max_amount_of_class = np.max(class_count)
    # print(f'Maximum amount of class: {max_amount_of_class}')

    x_new, y_new = [], []

    for label in unique_values:
        label_indices = np.where(y == label)[0]
        oversampled_version = np.random.choice(label_indices, max_amount_of_class, replace=True)

        x_new.extend(X[oversampled_version])        
        y_new.extend(y[oversampled_version])

    # print(f'New data: {len(x_new)}, {len(y_new)}')
    return np.array(x_new), np.array(y_new)        

def rebalanced_stratified_split(X_orig, y_orig, test_size=0.2, random_state=None):
    # Empty train and test sets
    X_train, X_test, y_train, y_test = [], [], [], []

    # Rebalance function implementation
    x_rb, y_rb = rebalance(X=X_orig, y=y_orig)
    # Initialization of the random state
    np.random.seed(random_state)
    unique_values, class_count = np.unique(y_rb, return_counts=True)
    # print(class_count)
    class_test_sizes = np.round(class_count * test_size).astype(int)
    # print(f'Class test size: {class_test_sizes}')

    for class_label, test_size in zip(unique_values, class_test_sizes):
        class_indices = np.where(y_rb == class_label)[0]
        np.random.shuffle(class_indices)
        
        test_indices = class_indices[:test_size]
        train_indices = class_indices[test_size:]
        # print(f'Class: {class_label}, test: {len(test_indices)}, train: {len(train_indices)}')

        X_train.extend(x_rb[train_indices])
        X_test.extend(x_rb[test_indices])
        y_train.extend(y_rb[train_indices])
        y_test.extend(y_rb[test_indices])
    
    # print(X_train[:15], X_test[:15], y_train[:15], y_test[:15])
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

# Question 3 call
# rebalanced_stratified_split(data_mtx, targets, test_size=0.2, random_state=42)

# Question 3 functions
def accuracy_score(y_true, y_pred):
    num_of_correct_preds = np.sum(y_true == y_pred)
    total_preds = len(y_true)
    return num_of_correct_preds / total_preds

def confusion_matrix(y_true, y_pred):
    unique_values = np.unique(y_true)
    num_of_values = len(unique_values)
    conf_matrix = np.zeros((num_of_values, num_of_values)) 
    # print(conf_matrix)

    for true_label, pred_label in zip(y_true, y_pred):
        true_index = np.where(unique_values == true_label)[0][0]
        pred_index = np.where(unique_values == pred_label)[0][0]
        conf_matrix[true_index, pred_index] += 1

    return conf_matrix.astype(int)
    
def predictive_performance_estimate(classifier, data_mtx, targets, test_size, n_rep=3):
    acc_scores = []
    for rep in range(n_rep):
        X_train, X_test, y_train, y_test = rebalanced_stratified_split(data_mtx, targets, test_size=test_size, random_state=rep)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        # print(f'Accuracy for repetition {rep}: {acc*100}%')
        # print(confusion_matrix(y_test, y_pred))
        acc_scores.append(acc)
    return np.mean(acc_scores), np.std(acc_scores)

# Question 4 functions
class LinearClassifier:
    def __init__(self, weights=None, bias=None, threshold=0.5):
        self.weights = weights
        self.bias = bias
        self.threshold = threshold
    
    def fit(self, X, y):
        # Add bias term to feature matrix
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        # Initialize weights if not provided
        if self.weights is None:
            self.weights = np.zeros(X_with_bias.shape[1])
        # Fit the model using the normal equation
        self.weights = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    
    def predict(self, X):
        # Add bias term to feature matrix
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        # Compute raw output scores
        raw_scores = X_with_bias @ self.weights
        # Apply threshold to raw scores to get binary predictions
        binary_predictions = (raw_scores >= self.threshold).astype(int)
        return binary_predictions

# Question 5 functions
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def predict_prob_single(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        sorted_indices = np.argsort(distances)
        k_nearest_labels = [self.y_train[i] for i in sorted_indices[:self.k]]
        unique_labels, label_counts = np.unique(k_nearest_labels, return_counts=True)
        probabilities = label_counts / self.k
        return probabilities
    
    def predict(self, X):
        predictions = [self.predict_prob_single(x) for x in X]
        predictions = [np.argmax(probabilites) for probabilites in predictions]
        return np.array(predictions)

# Question 6 functions
class CostSensitiveKNNClassifier:
    def __init__(self, k=3, class_cost_matrix=[[0, 1], [1, 0]]):
        self.k = k
        self.class_cost_matrix = np.array(class_cost_matrix)
        self.X_train = None
        self.y_train = None
        self.class_probabilities = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.class_probabilities = self._compute_class_probabilities()
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def predict_prob_single(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        sorted_indices = np.argsort(distances)
        k_nearest_labels = [self.y_train[i] for i in sorted_indices[:self.k]]
        unique_labels, label_counts = np.unique(k_nearest_labels, return_counts=True)
        probabilities = label_counts / self.k
        return probabilities
    
    def _compute_class_probabilities(self):
        class_probabilities = []
        for x in self.X_train:
            class_probabilities.append(self.predict_prob_single(x))
        return np.array(class_probabilities)
    
    def predict(self, X):
        predictions = []
        for x in X:
            probabilities = self.predict_prob_single(x)
            cost_vector = np.dot(probabilities, self.class_cost_matrix)
            predicted_class = np.argmin(cost_vector)
            predictions.append(predicted_class)
        return np.array(predictions)

# Question 7 functions
class GroupKNNClassifier:
    def __init__(self, k=3, groups=None, group_weights=None):
        self.k = k
        self.groups = groups
        self.group_weights = group_weights
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def distance(self, x):
        distances = []
        for x_train in self.X_train:
            weighted_distance = 0
            for group, weight in zip(self.groups, self.group_weights):
                group_distance = self.euclidean_distance(x[group], x_train[group])
                weighted_distance += group_distance * weight
            distances.append(weighted_distance)
        return distances
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = self.distance(x)
            sorted_indices = np.argsort(distances)
            k_nearest_labels = [self.y_train[i] for i in sorted_indices[:self.k]]
            unique_labels, label_counts = np.unique(k_nearest_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(label_counts)]
            predictions.append(predicted_label)
        return np.array(predictions)
    
# Question 8 functions
class AutoGroupsKNNClassifier:
    def __init__(self, k=3, param=0.5, weight=0.5):
        self.k = k
        self.param = param
        self.weight = weight
        self.X_train = None
        self.y_train = None
        self.groups = None
        self.group_weights = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.groups, self.group_weights = self._auto_group_features(X)
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = self._distance(x)
            sorted_indices = np.argsort(distances)
            k_nearest_labels = [self.y_train[i] for i in sorted_indices[:self.k]]
            unique_labels, label_counts = np.unique(k_nearest_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(label_counts)]
            predictions.append(predicted_label)
        return np.array(predictions)
    
    def _distance(self, x):
        distances = []
        for x_train in self.X_train:
            weighted_distance = 0
            for group, weight in zip(self.groups, self.group_weights):
                group_distance = np.linalg.norm(x[group] - x_train[group]) * weight
                weighted_distance += group_distance
            distances.append(weighted_distance)
        return distances
    
    def _auto_group_features(self, X):
        # Calculate feature importances
        feature_importances = np.mean(X, axis=0) / np.sum(np.mean(X, axis=0))
        # Group features based on the threshold parameter
        groups = [[i] for i in range(X.shape[1])]
        for i, importance in enumerate(feature_importances):
            if importance < self.param:
                groups[0].append(i)
            else:
                groups[1].append(i)
        # Assign weights to groups
        group_weights = [1 - self.weight, self.weight]
        return groups, group_weights


# _________________________________________________________________________________________
# Ready functions to call

#These functions are provided for you. You do not need to modify them in any way, just execute them.
def make_dataset_n_features_outliers(param):
    n_instances = 300
    fraction_of_outliers = 0.3
    n_informative_features = 2
    n_non_informative_features = param
    std = .75
    outliers_std = 2
    data_mtx, targets = make_dataset(n_instances, fraction_of_outliers, n_informative_features, n_non_informative_features, std, outliers_std)
    return data_mtx, targets

def make_dataset_outliers_std(param):
    n_instances = 1000
    fraction_of_outliers = 0.3
    n_informative_features = 2
    n_non_informative_features = 10
    std = 1
    outliers_std = param
    data_mtx, targets = make_dataset(n_instances, fraction_of_outliers, n_informative_features, n_non_informative_features, std, outliers_std)
    return data_mtx, targets

def plot_predictive_error_estimate_vs_param(make_dataset_func, classifier, params, n_rep=30, title='', xlabel=''):
    acc_means = []
    acc_stds = []
    for param in params:
        data_mtx, targets = make_dataset_func(param)
        # print(f'Data matrix and targets generated successfully')
        # print(f'Data matrix shape: {data_mtx.shape}, targets shape: {targets.shape}')   
        mean_acc, std_acc = predictive_performance_estimate(classifier, data_mtx, targets, test_size=.3, n_rep=n_rep)
        print(f'Param: {param}, mean accuracy: {mean_acc}, std accuracy: {std_acc}')
        acc_means.append(mean_acc)
        acc_stds.append(std_acc)
    acc_means = np.array(acc_means)
    acc_stds = np.array(acc_stds)

    plt.figure(figsize=(10,4))
    plt.plot(params, acc_means, lw=2, c='k')
    plt.fill_between(params, acc_means-acc_stds,acc_means+acc_stds,color='b', alpha=.2)
    plt.xlabel(xlabel)
    plt.ylim(0.5,1)
    plt.grid()
    plt.title(title)
    plt.show()


import time
start_time = time.time()

# params = np.arange(0, 150, 25)

# classifier = KNNClassifier(k=3)
# plot_predictive_error_estimate_vs_param(make_dataset_n_features_outliers, classifier, params, n_rep=30, title='KNN model performance decreases', xlabel='n_non_informative_features')


# classifier = AutoGroupsKNNClassifier(k=3,param=.5,weight=.1)
# plot_predictive_error_estimate_vs_param(make_dataset_n_features_outliers, classifier, params, n_rep=30, title='AutoGroupsKNN model performance does not decrease', xlabel='n_non_informative_features')


params = np.arange(2,20,4)

# classifier = LinearClassifier()
# plot_predictive_error_estimate_vs_param(make_dataset_outliers_std, classifier, params, n_rep=30, title='Linear model performance decreases', xlabel='outliers_std')

classifier = AutoGroupsKNNClassifier(k=3,param=.5,weight=.1)
plot_predictive_error_estimate_vs_param(make_dataset_outliers_std, classifier, params, n_rep=5, title='AutoGroupsKNN model performance does not decrease', xlabel='outliers_std')
print("Execution time:", round(time.time() - start_time, 2), "seconds")

save_history()