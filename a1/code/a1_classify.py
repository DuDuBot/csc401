import argparse
import os
import numpy as np
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.base import clone


classifiers = [SGDClassifier(), GaussianNB(),
                                 RandomForestClassifier(n_estimators=10, max_depth=5),
                                 MLPClassifier(alpha=0.05), AdaBoostClassifier()]


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return np.trace(C) / np.sum(C) if np.sum(C) > 0 else 0.


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    diag = np.diagonal(C)
    sum_by_class = np.sum(C, axis=1)
    return np.divide(diag, sum_by_class, out=np.zeros(len(diag)), where=sum_by_class!=0)


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    diag = np.diagonal(C)
    sum_by_classification = np.sum(C, axis=0)
    return np.divide(diag, sum_by_classification, out=np.zeros((len(diag))), where=sum_by_classification!=0)
    

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''

    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")
    iBest = 0
    best_acc = 0
    best_f1 = 0
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        for i, to_clone in enumerate(classifiers):
            cls = clone(to_clone)
            cls.fit(X_train, y_train)
            C = confusion_matrix(y_test, cls.predict(X_test))
            acc = accuracy(C)
            rec = recall(C)
            prec = precision(C)
            if acc > best_acc:
                best_acc = acc
                iBest = i
            outf.write(f'Results for {str(cls.__class__).split(".")[-1].replace(">", "")}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
            outf.write(f'\tConfusion Matrix: \n{C}\n\n')

    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")
    X_1k, y_1k = None, None
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        cls = clone(classifiers[iBest])
        for ds_amount in [1, 5, 10, 15, 20]:
            ds_amount = ds_amount * 1000
            print(f"performing ds_amount: {ds_amount}")
            # evenly select per class
            X = X_train[:ds_amount]
            y = y_train[:ds_amount]
            if X_1k is None and y_1k is None:
                X_1k = X_train
                y_1k = y_train
            train_set = shuffle(X, y, random_state=2)
            cls.fit(*train_set)
            C = confusion_matrix(y_test, cls.predict(X_test))
            outf.write(f'{ds_amount}: {accuracy(C):.4f}\n')

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # doing best 5 and best 50, and writing the pvalues.
        k_feat = 5
        selector = SelectKBest(f_classif, k_feat)
        X_new = selector.fit_transform(X_train, y_train)
        p_values = selector.p_values
        outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')
        k_feat = 50
        selector2 = SelectKBest(f_classif, k_feat)
        _ = selector2.fit_transform(X_train, y_train)
        p_values.append(selector2.p_values)
        outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')
        # training using best 5 features on 1k and 32k datasets for prev best cls.
        cls = clone(classifiers[i])
        cls.fit(X_new, y_train)
        C = confusion_matrix(y_test, cls.predict(X_test))
        accuracy_full = accuracy(C)
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        k_feat = 5
        selector = SelectKBest(f_classif, k_feat)
        X_new = selector.fit_transform(X_1k, y_1k)
        cls = clone(classifiers[i])
        cls.fit(X_new, y_1k)
        C = confusion_matrix(y_test, cls.predict(X_test))
        accuracy_1k = accuracy(C)
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')

        # outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        # outf.write(f'Top-5 at higher: {top_5}\n')
        pass


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
        '''
    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # for each fold:
        #     outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        # outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')
        pass


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    data = np.load(args.input)
    data = data[data.files[0]]
    best_accuracy =[]
    X_train, X_test, y_train, y_test = train_test_split(data[:, :173], data[:, -1],
                                                        test_size=0.2,
                                                        random_state=0,
                                                        stratify=data[:, -1]
                                                        )
    X_train, y_train = shuffle(X_train, y_train, random_state=2)
    X_test, y_test = shuffle(X_test, y_test, random_state=2)
    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    # TODO: load data and split into train and test.
    # TODO : complete each classification experiment, in sequence.
