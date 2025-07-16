import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class encoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        return self

    
    def transform(self, X):
        transformed = X.copy()
        for i in X.columns:
            if X[i].nunique()<len(X[i])/5:
                if X[i].nunique()>5:
                    enc = LabelEncoder()
                    transformed[i] = enc.fit_transform(X[i])
                
                else:
                    enc = OneHotEncoder(sparse_output=False)
                    transformed[i] = enc.fit_transform(X[[i]])
                    temp_X = pd.DataFrame(OneHotEncoder(drop='first', handle_unknown='ignore').fit_transform(X[[i]]).toarray())
                    transformed = pd.concat([transformed, temp_X], axis=1)
                    transformed.drop([i], axis=1, inplace=True)
            else:
                transformed.drop([i], axis=1, inplace=True)
        return transformed


class Imputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self,X):
        return self
    
    def transform(self,X):
        imputer = SimpleImputer(strategy='most_frequent')
        self.X = imputer.fit_transform(X)
        self.X = pd.DataFrame(self.X)
        return self.X


def Cleaner(df):
    # df = pd.read_csv("titanic.csv")
    # print(df.head(5))
    X,y = df.iloc[:,:-1],df.iloc[:,-1]

    num_col = X.select_dtypes(include='number')
    cat_col = X.select_dtypes(include='object')

    num_features = X.select_dtypes(include='number').columns
    cat_features = X.select_dtypes(include='object').columns

    #encd= encoder()
    #X2 = encd.fit_transform(cat_col)
    #print(X2)

    num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
    ])

    #num_transformed = num_pipeline.fit_transform(num_col)
    #print(num_transformed)

    cat_pipeline = Pipeline([
    ('imputer', Imputer()),
    ('encoder', encoder()),

    ])

    #cat_transformed = cat_pipeline.fit_transform(cat_col)
    #print (cat_transformed)

    from sklearn.compose import ColumnTransformer
    data_pipeline = ColumnTransformer([
                        ('numerical', num_pipeline, num_features),
                        ('categorical', cat_pipeline, cat_features)
    ])

    processed_X = pd.DataFrame(data_pipeline.fit_transform(X))
    
    data = pd.concat([processed_X, y], axis=1)

    data.dropna(axis=0, how='any', inplace=True)

    return data