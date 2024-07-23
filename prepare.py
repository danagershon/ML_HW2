import sklearn

## Normalizers

def StandardNormalize(df, df_new, columns):
  scaler = sklearn.preprocessing.StandardScaler()
  scaler.fit(df[columns])
  df_new[columns] = scaler.transform(df_new[columns])

def MinMaxNormalize(df, df_new, columns):
  scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
  scaler.fit(df[columns])
  df_new[columns] = scaler.transform(df_new[columns])


### Prepare Data
def prepare_data(training_data, new_data):
  prepared_data = new_data.copy()
  training_data_copy = training_data.copy()

  #Replace NaN values for household_income:
  median = training_data["household_income"].median()
  prepared_data["household_income"] = prepared_data["household_income"].fillna(median)

  #Prepare Blood Types
  prepared_data["SpecialProperty"] = prepared_data["blood_type"].isin(["O+", "B+"])
  prepared_data = prepared_data.drop("blood_type", axis=1)
  
  #Normalization
  StandardNormalize(training_data_copy, prepared_data, ["PCR_01", "PCR_02", "PCR_05", "PCR_06", "PCR_07", "PCR_08"])
  MinMaxNormalize(training_data_copy, prepared_data, ["PCR_10", "PCR_03", "PCR_04", "PCR_09"])
  return prepared_data
