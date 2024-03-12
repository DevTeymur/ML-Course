import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
from submission_utils import save_history, check_and_prepare_for_submission
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def make_positive_data(n_instances, fraction_of_outliers, n_informative_features, n_non_informative_features, std, outliers_std):
    # Calculation of the number of outliers and inliners
    n_inliners = int((1 - fraction_of_outliers) * n_instances)
    n_outliers = int(fraction_of_outliers * n_instances)
    # print(f'Normal: {n_inliners}, outliers {n_outliers}')

    # Generate inliners
    informative_inliners = np.random.normal(1, std, (n_inliners, n_informative_features))
    non_informative_inliners = np.random.normal(0, std, (n_inliners, n_non_informative_features))

    # print(f'Informative inliners: {informative_inliners.shape}, non-informative inliners: {non_informative_inliners.shape}')
    # print(f'{informative_inliners[:5]}')
    # print(f'{non_informative_inliners[:5]}')

    # Generate outliers
    informative_outliers = np.random.normal(-1, outliers_std, (n_outliers, n_informative_features))
    non_informative_outliers = np.random.normal(0, outliers_std, (n_outliers, n_non_informative_features))
    # print(f'Informative outliers: {informative_outliers.shape}, non-informative outliers: {non_informative_outliers.shape}')
    # print(f'{informative_outliers[:5]}')
    # print(f'{non_informative_outliers[:5]}')

    # Join all generated data
    all_positive_data = np.concatenate((np.hstack((informative_inliners, non_informative_inliners)),
                                    np.hstack((informative_outliers, non_informative_outliers))))
    # print(f'All data: {all_positive_data.shape}')
    # print(f'All data: {type(all_positive_data)}')
    return all_positive_data

def make_negative_data(n_instances, n_informative_features, n_non_informative_features, std):
    informative = np.random.normal(-1, std, (n_instances, n_informative_features))
    non_informative = np.random.normal(0, std, (n_instances, n_non_informative_features))
    # print(f'Informative: {informative.shape}, non-informative: {non_informative.shape}')
    # print(f'{informative[:5]}')
    # print(f'{non_informative[:5]}')

    all_negative_data = np.hstack((informative, non_informative))
    # print(all_negative_data.shape)
    # print(all_negative_data[:5])
    # print(type(all_negative_data))
    return all_negative_data
    
def make_dataset(n_instances, fraction_of_outliers, n_informative_features, n_non_informative_features, std, outliers_std):
    amount_of_positives = n_instances // 2
    amount_of_negatives = n_instances - amount_of_positives
    positive_data = make_positive_data(n_instances=amount_of_positives, 
                                       fraction_of_outliers=fraction_of_outliers, 
                                       n_informative_features=n_informative_features, 
                                       n_non_informative_features=n_non_informative_features, 
                                       std=std, 
                                       outliers_std=outliers_std)
    negative_data = make_negative_data(n_instances=amount_of_negatives, 
                                       n_informative_features=n_informative_features, 
                                       n_non_informative_features=n_non_informative_features, 
                                       std=std)
    
    data_mtx = np.concatenate((positive_data, negative_data))
    targets = np.concatenate((np.ones(amount_of_positives), np.zeros(amount_of_negatives)))
    # print(f'Final data: {data_mtx.shape}')
    # print(f'Targets: {targets.shape}')
    # print(data_mtx[:5]), print(targets[:5])
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
        plt.scatter(data_mtx[:, 0], data_mtx[:, 1], data_mtx[:, 2])
    else:
        plt.scatter(data_mtx[:, 0], data_mtx[:, 1], data_mtx[:, 2], c=targets)
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

data_mtx, targets = make_dataset(n_instances, fraction_of_outliers, n_informative_features, n_non_informative_features, std, outliers_std)

# plot2d(data_mtx, targets, title='Data 2D', size=8)
# plot3d(data_mtx, targets, title='Data 3D', size=8)


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
    print(class_count)
    class_test_sizes = np.round(class_count * test_size).astype(int)
    print(f'Class test size: {class_test_sizes}')

    for class_label, test_size in zip(unique_values, class_test_sizes):
        class_indices = np.where(y_rb == class_label)[0]
        np.random.shuffle(class_indices)
        
        test_indices = class_indices[:test_size]
        train_indices = class_indices[test_size:]
        print(f'Class: {class_label}, test: {len(test_indices)}, train: {len(train_indices)}')

        X_train.extend(x_rb[train_indices])
        X_test.extend(x_rb[test_indices])
        y_train.extend(y_rb[train_indices])
        y_test.extend(y_rb[test_indices])
    
    print(X_train[:15], X_test[:15], y_train[:15], y_test[:15])
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

rebalanced_stratified_split(data_mtx, targets, test_size=0.2, random_state=42)


def accuracy_score(y_true, y_pred):
    num_of_correct_preds = np.sum(y_true == y_pred)
    total_preds = len(y_true)
    return num_of_correct_preds / total_preds

def confusion_matrix(y_true, y_pred):
    unique_values = np.unique(y_true)
    num_of_values = len(unique_values)
    conf_matrix = np.zeros((num_of_values, num_of_values)) 
    print(conf_matrix)

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
        print(f'Accuracy for repetition {rep}: {acc*100}%')
        print(confusion_matrix(y_test, y_pred))
        acc_scores.append(acc)
    return np.mean(acc_scores), np.std(acc_scores)

class LinearClassifier:
    def __init__(self, weights=None, bias=None, threshold=0.5):
        self.weights = None
        self.bias = None
        self.threshold = threshold
    
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def eucledean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def predict_prob_single(self, x):
        pass

    def predict(self, X_test):
        return [self.predict_prob_single(x) for x in X_test]

save_history()