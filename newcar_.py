import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv('Cars_24.csv')

df.head()

df.info()

df.shape

df =  df.drop(columns=['5'] , axis=1)

df.isna().sum()

# splitting

x = df.drop(columns=['selling_price'] , axis=1)
y = df['selling_price']

x.shape , y.shape

from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain, ytest = train_test_split(x , y ,test_size=0.2 , random_state=42)

xtrain.shape ,ytrain.shape

xtest.shape , ytest.shape

Pipeline = Pipeline([
    ('encoder' , TargetEncoder(cols=['make','model'])),
    ('poly' , PolynomialFeatures(degree=3)),
    ('scaller' , StandardScaler()),
    ('medel' , Ridge(alpha=1000))
])

Pipeline.fit(xtrain , ytrain)

print('traing score' , Pipeline.score(xtrain , ytrain))
print('test score' , Pipeline.score(xtest , ytest))

joblib.dump(Pipeline , 'car_price_model.joblib')





