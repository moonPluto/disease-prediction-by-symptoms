import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
df=pd.read_csv("csvfile/s2d.csv")
xc=df.drop("Disease",axis=1)
tg=df["Disease"]
le=LabelEncoder()
yb=le.fit_transform(tg)
xti,xts,yti,yts=train_test_split(xc,yb,test_size=0.2,random_state=42)
print('data spilited in train and test set...')
model=RandomForestClassifier()
model.fit(xti,yti)
print('model train finish...')
ypred=model.predict(xts)
acc=accuracy_score(yts,ypred)
print('model accuracy is =',acc)
pickle.dump(model,open("store/rfc.pkl","wb"))
print('model saved as pkl file.')

