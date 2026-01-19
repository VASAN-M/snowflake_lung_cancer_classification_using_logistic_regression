

import pandas as pd
import snowflake.connector
conn=snowflake.connector.connect(
    user='your_username',
    password='your_password',
    account='acc_id',
    database='db_name',
    schema='PUBLIC',
    warehouse='COMPUTE_WH'
)

query='SELECT * FROM "table_name"'

df=pd.read_sql(query,conn)
conn.close()

print(df.head())
print(df.info())

for col in df.columns:
  print(df[col].name,df[col].unique())

y = df["LUNG_CANCER"].map({True:1, False:0})
x = df.drop("LUNG_CANCER", axis=1)

x["GENDER"] = x["GENDER"].map({"M":1, "F":0})

from sklearn.preprocessing import StandardScaler
num_col=df.select_dtypes(include=['int','float']).columns
scaler=StandardScaler()
x[num_col]=scaler.fit_transform(x[num_col])

for col in x.columns:
  print(x[col].name,x[col].unique())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,roc_auc_score

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=36,stratify=y)
model=LogisticRegression(max_iter=1000,class_weight="balanced")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_prob = model.predict_proba(x_test)[:,1]
print(classification_report(y_test,y_pred))
print(roc_auc_score(y_test,y_prob))

import joblib
joblib.dump(model,'logisticregr.joblib')
joblib.dump(scaler,'scaler.joblib')

