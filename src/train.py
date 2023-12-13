import streamlit as sl
import pandas as pd
from kmodes.kprototypes import KPrototypes
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import joblib

csv_file = '.\shopping_behavior_updated.csv'
df = pd.read_csv(csv_file)

sl.write("# Shopping Behavior Trend")
sl.table(df.head())

drop_columns = [df.columns[0]] + list(df.columns[5:9]) + list(df.columns[11:18])

df = df.drop(columns=drop_columns)

sl.table(df.head())

score = []
categorical_columns = list(range(1, 5))
for k in tqdm(range(2, 10)):
  model = KPrototypes(n_clusters=k)
  model.fit(df, categorical=categorical_columns)
  score.append(model.cost_)
plt.plot(range(2, 10), score)

model = KPrototypes(n_clusters=4)
member = model.fit_predict(df, categorical=categorical_columns)

joblib.dump(model, 'model1.joblib')

df_pred = df.copy()
df_pred["cluster"] = member #memasukkan kelas klaster ke masing-masing data dengan menambahkan kolom klaster
df_pred.to_csv('output_file.csv', index=False)
print(df_pred.head())

centroids = model.cluster_centroids_

df_centroids = pd.DataFrame(centroids, columns=[
    'Age', 'Review Rating',
    'Gender', 'Item Purchased',
    'Category', 'Season'
])
df_centroids['Age'] = df_centroids['Age'].astype(float).round().astype(int)
print(df_centroids)