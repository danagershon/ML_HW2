import unittest
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from SoftSVM import SoftSVM
from verify_gradients import compare_gradients
from visualize_clf import visualize_clf

class TestSoftSVM(unittest.TestCase):

    def setUp(self) -> None:
        # load train annd test sets data from csv files
        self.train_dataset = pd.read_csv('expected_train_prepared.csv')
        self.test_dataset = pd.read_csv('expected_test_prepared.csv')
        
        # create dataframe for feature pcr_04 and PCR_10 only, to predict the spread label
        self.df_train_pcr_04_09 = self.train_dataset[['PCR_04', 'PCR_09']]
        self.df_train_spread_labels = self.train_dataset['spread']

        self.X_train, self.y_train = self.df_train_pcr_04_09.values, self.df_train_spread_labels.values

        self.df_test_pcr_04_09 = self.test_dataset[['PCR_04', 'PCR_09']]
        self.df_test_spread_labels = self.test_dataset['spread']

    def test_compare_gradients(self):
        compare_gradients(self.X_train, self.y_train, deltas=np.logspace(-5, 0, 10))

    def test_lr_in_range(self):        
        for lr in np.logspace(-11, -3, 5):
            C = 1e5
            clf = SoftSVM(C=C, lr=lr)
            X_train_poly = PolynomialFeatures(degree=3,).fit_transform(self.X_train)
            X_train_poly = MinMaxScaler(feature_range=(-1,1)).fit_transform(X_train_poly)
            losses, accuracies = clf.fit_with_logs(X_train_poly, self.y_train, max_iter=5000)
            plt.figure(figsize=(13, 6))
            plt.subplot(121), plt.grid(alpha=0.5), plt.title ("Training Loss")
            plt.semilogy(losses), plt.xlabel("Step"), plt.ylabel("Loss")
            plt.subplot(122), plt.grid(alpha=0.5), plt.title ("Training Accuracy")
            plt.plot(accuracies), plt.xlabel("Step"), plt.ylabel("Accuracy")
            plt.tight_layout()
            print(f"lr={lr}")
            plt.show()

    def test_chosen_lr_svm(self):
        svm_clf = Pipeline([('feature_mapping', PolynomialFeatures(degree=3,)),
                            ('scaler', MinMaxScaler()),
                            ('SVM', SoftSVM(C=1e5, lr=1e-7))])
        svm_clf.fit(self.X_train, self.y_train)

        visualize_clf(svm_clf, self.df_train_pcr_04_09, self.df_train_spread_labels, 
                      title='chosen_lr_SVM decision regions', xlabel='PCR_04', ylabel='PCR_09')

        train_spread_pred = svm_clf.predict(self.df_train_pcr_04_09.values)
        test_spread_pred = svm_clf.predict(self.df_test_pcr_04_09.values)

        print(f"Accuracy Train: {accuracy_score(self.df_train_spread_labels, train_spread_pred)*100}%")
        print(f"Accuracy Test: {accuracy_score(self.df_test_spread_labels, test_spread_pred)*100}%")

    def test_svm_rbf_gamma_values(self):
        def do_SVC_gamma(gamma):
            model = SVC(C=1.0, kernel='rbf', gamma=gamma)
            model.fit(self.X_train, self.y_train)
            visualize_clf(model, self.df_train_pcr_04_09, self.df_train_spread_labels, title=f'SVM_RBF(gamma={gamma}) decision regions', xlabel='PCR_04', ylabel='PCR_09')
            train_spread_pred = model.predict(self.X_train)
            test_spread_pred = model.predict(self.df_test_pcr_04_09.values)
            print(f"Accuracy Train: {accuracy_score(self.df_train_spread_labels, train_spread_pred)*100}%")
            print(f"Accuracy Test: {accuracy_score(self.df_test_spread_labels, test_spread_pred)*100}%")

        do_SVC_gamma(1e-7)
        do_SVC_gamma(200)
        do_SVC_gamma(5000)