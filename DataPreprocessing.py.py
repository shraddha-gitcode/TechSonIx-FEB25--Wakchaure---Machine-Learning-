import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv('car_dataset_india.csv')

# Handling missing values
# For categorical data (Service_Cost, Price)
cat_imputer = SimpleImputer(strategy='most_frequent')
df[['Service_Cost', 'Price']] = cat_imputer.fit_transform(df[['Service_Cost', 'Price']])

# For numerical data (if any numerical columns present)
# num_imputer = SimpleImputer(strategy='mean')
# df[['NumericalColumn']] = num_imputer.fit_transform(df[['NumericalColumn']])

# Normalize numerical features (if any numerical columns present)
# scaler = StandardScaler()
# df[['NumericalColumn']] = scaler.fit_transform(df[['NumericalColumn']])

# Encoding categorical features
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df[['Service_Cost']]).toarray()
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Service_Cost']))

# Merge the encoded features back to the main dataframe
df = df.drop('Service_Cost', axis=1)
df = pd.concat([df, encoded_df], axis=1)

print(df.head())
