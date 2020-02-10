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
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.base import clone
import time

classifiers = [SGDClassifier(), GaussianNB(),
               RandomForestClassifier(n_estimators=10, max_depth=5),
               MLPClassifier(alpha=0.05), AdaBoostClassifier()]

classifiers_bonus = [SGDClassifier(random_state=2), GaussianNB(),
                     RandomForestClassifier(n_estimators=10, max_depth=5, random_state=2),
                     MLPClassifier(alpha=0.05, random_state=2), AdaBoostClassifier(random_state=2),
                     MultinomialNB()
                     ]

search_params = {'SGDClassifier':
                   {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'alpha': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
                    'l1_ratio': [0.1, 0.15, 0.2],
                    'max_iter': [100, 1000, 10000],
                    'average': [False, 10, 100, 1000],
                    'early_stopping': [False, True],
                    },
                 'GaussianNB': {'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]},
                 'RandomForestClassifier':
                   {'n_estimators': [10, 20, 30, 40, 50, 75, 100],
                    'max_depth': [3, 5, 10, 20, 30],
                    'min_samples_split': [1, 2, 3, 5],
                    'min_samples_leaf': [1, 2]},
                 'MLPClassifier':
                   {'hidden_layer_sizes': [1, 5, 10, 20, 50, 100],
                    'activation': ['relu', 'tanh'],
                    'learning_rate': ['constant', 'adaptive'],
                    'max_iter': [200, 500, 1000],
                    'early_stopping': [False, True]},
                 'AdaBoostClassifier':
                   {'n_estimators': [25, 50, 75, 100],
                    'learning_rate': [0.5, 1, 1.5]},
                 'MultinomialNB': {'alpha': [0.00001, 0.5, 1, 1.5, 2]}}


def accuracy(C):
  ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
  return np.trace(C) / np.sum(C) if np.sum(C) > 0 else 0.


def recall(C):
  ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
  diag = np.diagonal(C)
  sum_by_class = np.sum(C, axis=1)
  return np.divide(diag, sum_by_class, out=np.zeros(len(diag)),
                   where=sum_by_class != 0)


def precision(C):
  ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
  diag = np.diagonal(C)
  sum_by_classification = np.sum(C, axis=0)
  return np.divide(diag, sum_by_classification, out=np.zeros((len(diag))),
                   where=sum_by_classification != 0)


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
      name = str(cls.__class__).split(".")[-1].replace(">", "").replace("\'",
                                                                        "")
      outf.write(f'Results for {name}:\n')  # Classifier name
      outf.write(f'\tAccuracy: {acc:.4f}\n')
      outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
      outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
      outf.write(f'\tConfusion Matrix: \n{C}\n\n')
    outf.write('we see that the RandomForestClassifier, MLP, and the AdaBoost '
               'classifiers perform the best on this data. RandomForest and AdaBoost are '
               'expected because these are both ensemble-based models '
               '(they aggregrate many weak-learners to reduce variance).'
               ' The reduced helps these models generalize better, especially '
               'in low data regimes. However, the MLP model, being a neural architecture, is a surprising result as they generally require lots of data to perform well. The MLP structure chosen by SKLearn is likely a simple MLP that does not require much data. '
               'The SGDClassifier uses a linear SVM given the default configuration which is surprising that it does not perform well, since they generally perform well despite low-data (since they learn a simple margin from support vectors). Finally, the GaussianNB likely suffers '
               'despite NB\'s relative success in simple NKP problems because '
               'text is oftern not modelled well using a Gaussian distribution.')
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
      # evenly select per class is done thanks to stratify
      X = X_train[:ds_amount]
      y = y_train[:ds_amount]
      if X_1k is None and y_1k is None:
        X_1k = X_train
        y_1k = y_train
      train_set = shuffle(X, y, random_state=2)
      cls.fit(*train_set)
      C = confusion_matrix(y_test, cls.predict(X_test))
      outf.write(f'{ds_amount}: {accuracy(C):.4f}\n')
    outf.write('Evidently, the more data has a significant impact on the'
               'model performance, with a strict increase in the test '
               'accuracy as the training data increases. Interestingly, '
               'There is not much improvement between 15k to 20k samples, '
               'likely because we are hitting the limit capacity for the'
               'default 50 classifiers in the Adaboost classifier (best).'
               'Regardless, the accuracies are quite low across all dataset'
               'sizes, likely due to the variance captured in the training '
               'data. Generally, these descriptive features like age of '
               'acquisition, etc., might not well relate with the actual '
               'categorization of left, right, alt, and center. An improvement'
               'might be to use a higher-capacity model, with lots more raw data. '
               'I would suggest raw data because, with enough of it, and a '
               'good learning algorithm, the model can learn from the text'
               'the important features, rather than us guessing what will '
               'explain the correlation best. Of course, we could also try '
               'other features that are better tailored to enable the models '
               'to learn the correlation between input and output.')
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
    features_32k = selector.get_support(True)
    p_values = selector.pvalues_
    outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')
    k_feat = 50
    selector2 = SelectKBest(f_classif, k_feat)
    _ = selector2.fit_transform(X_train, y_train)
    p_values = selector2.pvalues_
    outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')
    # training using best 5 features on 1k and 32k datasets for prev best cls.
    cls = clone(classifiers[i])
    cls.fit(X_new, y_train)  # 32k dataset
    C = confusion_matrix(y_test, cls.predict(selector.transform(X_test)))
    accuracy_full = accuracy(C)
    outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
    k_feat = 5
    selector = SelectKBest(f_classif, k_feat)
    X_new = selector.fit_transform(X_1k, y_1k)
    # print(f'1k p-values: {[round(pval, 4) for pval in selector.pvalues_]}')
    cls = clone(classifiers[i])
    try:
      cls.set_params(random_state=8)
    except:
      pass
    cls.fit(X_new, y_1k)
    C = confusion_matrix(y_test, cls.predict(selector.transform(X_test)))
    accuracy_1k = accuracy(C)
    outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
    features_1k = selector.get_support(True)
    intersection = np.intersect1d(features_1k, features_32k,
                                  return_indices=True)
    top_5 = features_32k
    outf.write(f'Chosen feature intersection: {intersection}\n')
    outf.write(f'Top-5 at higher: {top_5}\n')
    outf.write('(a) and (c) The top-5 features for both the 32k and 1k datasets are '
               'the same, with them being the number of first-person '
               'pronouns, second-person pronouns, number of adverbs, as well as 2 features from the'
               ' linquistic query and word count features. The linquistic '
               'features are supposed to be useful for determining sentiment, '
               'which is likely a differentiating factor between more left- versus'
               ' more right-wing people (and therefore their statements). Since '
               'left-wing people are more progessive, it is possible that is '
               'captured in the way they speak. Further, looking at the mean'
               'values for pronouns, the first-person pronouns has a mean'
               'value of ~2.38 for left-leaning texts and values of ~1 for the '
               'other labels, indicating that left texts use substantially more'
               'first-person pronouns. Second-person pronounds have a '
               'good distribution between all 4 labels, ranging from ~1.5'
               'for left texts to ~0.3 for right texts, and being sometwhere'
               ' in the middle for center and alt texts. These (somewhat)'
               ' separate values indicate that these features are good for '
               'helping diffentiate the class for a text. The standard '
               'deviation is relatively high, (>2), indicating that this feature'
               'does not linearly separate our data (which we already knew, '
               'or our accuracies would be much higher). However, it is still useful.'
               'The mean number of adverbs ranges from 1.5 to 4.5, with left and right '
               'texts showing more adverbs than center of alt. This result is intersting'
               ', as it is possible that left or right texts use more adverbs to '
               'express stronger emotion. The standard deviation goes as high as 7, '
               'though, indicating that this ranges a lot (as it should).'
               ' (b) The pvalues generally lower if we are given more data, as with more samples, we are genenerally more confidence in our results from a statistical test. This intuition would result in a lower pvalue, as expected.')


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
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    validator = KFold(shuffle=True, random_state=2)
    cls_to_acc = {}
    names = []
    best_name = None
    j = 0
    p_values = []
    for cls_base in classifiers:
      name = str(cls_base.__class__).split(".")[-1].replace(">", "").replace(
        "\'",
        "")
      print(f'working on classifier: {name}')
      kfold_accuracies = []
      fold = 0
      for train_index, test_index in validator.split(X_all):
        print(f'fold: {fold}')
        cls = clone(cls_base)
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]
        cls.fit(X_train, y_train)
        C = confusion_matrix(y_test, cls.predict(X_test))
        kfold_accuracies.append(accuracy(C))
        fold += 1
      outf.write(
        f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
      cls_to_acc[name] = kfold_accuracies
      if j == i:
        best_name = name
      else:
        names.append(name)
      j += 1
    for name in names:
      S, pvalue = ttest_rel(cls_to_acc[name], cls_to_acc[best_name])
      p_values.append(pvalue)
    outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')


def classify_bonus(output_dir, X_train, X_test, y_train, y_test):
  # do just the base classifiers, but with scaling.
  with open(f"{output_dir}/a1_bonus.txt", 'w') as outf:
    scaler = StandardScaler()
    scaler.fit(X_train)
    iBest = 0
    best_acc = 0
    outf.write('Trying mean and std removal scaling\n')
    for i, to_clone in enumerate(classifiers_bonus):
      cls = clone(to_clone)
      name = str(cls.__class__).split(".")[-1].replace(">", "").replace("\'",
                                                                        "")
      print(f'starting classifier: {name} for standard scaling')
      outf.write(f'Results for {name}:\n')  # Classifier name
      if name.lower().find('multi') == -1:
        cls.fit(scaler.transform(X_train), y_train)
        C = confusion_matrix(y_test, cls.predict(scaler.transform(X_test)))
      else:
        outf.write('performing multinomialNB without scaling (because you cannot.\n')
        cls.fit(X_train.clip(min=0), y_train)
        C = confusion_matrix(y_test, cls.predict(X_test.clip(min=0)))
      acc = accuracy(C)
      rec = recall(C)
      prec = precision(C)
      if acc > best_acc:
        best_acc = acc
        iBest = i
      outf.write(f'\tAccuracy: {acc:.4f}\n')
      outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
      outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
      outf.write(f'\tConfusion Matrix: \n{C}\n\n')
    outf.write('\n')
    outf.write('Trying minmax scaling\n')
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    iBest = 0
    best_acc = 0
    for i, to_clone in enumerate(classifiers_bonus):
      cls = clone(to_clone)
      name = str(cls.__class__).split(".")[-1].replace(">", "").replace("\'",
                                                                        "")
      print(f'starting classifier: {name}  for minmax scaling')
      outf.write(f'Results for {name}:\n')  # Classifier name
      if name.lower().find('multi') == -1:
        cls.fit(scaler.transform(X_train), y_train)
        C = confusion_matrix(y_test, cls.predict(scaler.transform(X_test)))
      else:
        outf.write('performing multinomialNB without scaling (because you cannot.\n')
        cls.fit(X_train.clip(min=0), y_train)
        C = confusion_matrix(y_test, cls.predict(X_test.clip(min=0)))
      acc = accuracy(C)
      rec = recall(C)
      prec = precision(C)
      if acc > best_acc:
        best_acc = acc
        iBest = i
      outf.write(f'\tAccuracy: {acc:.4f}\n')
      outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
      outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
      outf.write(f'\tConfusion Matrix: \n{C}\n\n')
    outf.write('\n')
    outf.write('trying HyperParam optimization\n. As neither scaling preprocessing had a major effect, we will not use them.\n')
    iBest = 0
    cls_best = None
    best_acc = None
    for i, to_clone in enumerate(classifiers_bonus):  # now we use rgridsearch to do hyperparam optimization
      # randomized is always better than pure grid search in convergence. We could do bayesian optimization, but that takes
      # significantly more effort to set up with the Sklearn ecosystem (even using skopt).
      if i < 3:
        continue
      cls = clone(to_clone)
      name = str(cls.__class__).split(".")[-1].replace(">", "").replace("\'",
                                                                        "")
      print(f'starting Randomized search optimization for {name}\n')
      params = search_params[name]
      rgridsearch = RandomizedSearchCV(cls, params, n_iter=10, random_state=2, scoring='accuracy')
      outf.write(f'Results for {name}:\n')  # Classifier name
      if name.lower().find('multi') == -1:
        rgridsearch.fit(X_train, y_train)
        C = confusion_matrix(y_test, rgridsearch.predict(X_test))
      else:
        outf.write('performing multinomialNB without scaling (because you cannot.\n')
        rgridsearch.fit(X_train.clip(min=0), y_train)
        C = confusion_matrix(y_test, rgridsearch.predict(X_test.clip(min=0)))
      acc = accuracy(C)
      rec = recall(C)
      prec = precision(C)
      if acc > best_acc:
        best_acc = acc
        iBest = i
        cls_best = rgridsearch
      outf.write(f'\tAccuracy: {acc:.4f}\n')
      outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
      outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
      outf.write(f'\tConfusion Matrix: \n{C}\n\n')
      name = str(cls_best.estimator.__class__).split(".")[-1].replace(">", "").replace("\'",
                                                                        "")
      outf.write(f'params for this best classifier, "{name}" were: {cls_best.best_params_}\n')
    outf.write('\n')
    outf.write('As we can see, there is a significant increase of around 3-5\% per classifier. '
               'Unfortunately, the best classifier, AdaBoost, did not have a significant increase due to hyperparameter optimization.'
               'This hsows that though hyperparameter optimization does have a big impact, in this case, '
               'we are limited not by the model capacity, but by the feature information. It is possible that LSA or LDA, as we will explore next, '
               'will enable us to have a better accuracy.\n')


def classify_lda(output_dir, X_train, X_test, y_train, y_test):
  with open(f"{output_dir}/a1_bonus_class_lda.txt", 'w') as outf:
    cls = AdaBoostClassifier(**{'n_estimators': 100, 'learning_rate': 0.5})  # best classifier from rgridsearch
    cls.fit(X_train, y_train)
    C = confusion_matrix(y_test, cls.predict(X_test))
    outf.write(f"accuracy using LDa is: {accuracy(C)} for Adaboost, the best classifier from before.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", help="the input npz file from Task 2",
                      required=True)
  parser.add_argument(
    "-o", "--output_dir",
    help="The directory to write a1_3.X.txt files to.",
    default=os.path.dirname(os.path.dirname(__file__)))
  args = parser.parse_args()
  data = np.load(args.input)
  data = data[data.files[0]]
  best_accuracy = []
  stime = time.clock()
  # X_train, X_test, y_train, y_test = train_test_split(data[:, :173],
  #                                                     data[:, -1],
  #                                                     test_size=0.2,
  #                                                     random_state=0,
  #                                                     stratify=data[:, -1]
  #                                                     )
  # X_train, y_train = shuffle(X_train, y_train, random_state=2)
  # iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
  # print(f'done class31 at {time.clock()-stime}')
  # (X_1k, y_1k) = class32(args.output_dir, X_train, X_test, y_train, y_test,
  #                        iBest)
  # print(f'done class32 at {time.clock()-stime}')
  # class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
  # print(f'done class33 at {time.clock()-stime}')
  # class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)
  # print(f'done class34 at {time.clock()-stime}')
  # BELOW IS FOR BONUS, uncomment to run
  # print('starting bonus, this might take awhile')
  # classify_bonus(args.output_dir, X_train, X_test, y_train, y_test)
  # print(f'done class_bonus at {time.clock()-stime}')
  infile = args.input
  if infile.find('.npz') == -1:
    infile += '_bonus_LDA'
    outf = infile
  else:
    outf = infile[:infile.rfind('.')] + '_bonus_LDA' + '.txt'
    infile = infile[:infile.rfind('.')] + '_bonus_LDA' + infile[infile.rfind('.'):]
  data = np.load(infile)
  data = data[data.files[0]]
  best_accuracy = []
  stime = time.clock()
  X_train, X_test, y_train, y_test = train_test_split(data[:, :-1],
                                                      data[:, -1],
                                                      test_size=0.2,
                                                      random_state=0,
                                                      stratify=data[:, -1]
                                                      )
  X_train, y_train = shuffle(X_train, y_train, random_state=2)
  classify_lda(args.output_dir, X_train, X_test, y_train, y_test)
