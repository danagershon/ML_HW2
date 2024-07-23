import unittest
from sklearn.model_selection import train_test_split
import pandas as pd
import filecmp

from prepare import prepare_data

class TestPrepareData(unittest.TestCase):

    def setUp(self) -> None:
        # load data from csv
        self.dataset = pd.read_csv('virus_data.csv')
        # split to train and test sets
        self.df_train, self.df_test = train_test_split(self.dataset, train_size=0.8, random_state=74+40)

    def test_from_notebook(self):
        # Prepare training set according to itself
        train_df_prepared = prepare_data(self.df_train, self.df_train)

        # Prepare test set according to the raw training set
        test_df_prepared = prepare_data(self.df_train, self.df_test)

        outputPath = "train_prepared.csv"
        train_df_prepared.to_csv(outputPath)

        self.assertTrue(filecmp.cmp('expected_' + outputPath, outputPath), 'prepared train csv is differnce from expected')

        outputPath = "test_prepared.csv"
        test_df_prepared.to_csv(outputPath)

        self.assertTrue(filecmp.cmp('expected_' + outputPath, outputPath), 'prepared test csv is differnce from expected')
