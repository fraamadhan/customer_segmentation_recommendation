import pandas as pd
from kmodes.kprototypes import KPrototypes
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler

csv_file = '..\data\shopping_behavior_updated.csv'
df = pd.read_csv(csv_file)

spring_data = df[df["Season"] == "Spring"].copy()
summer_data = df[df["Season"] == "Summer"].copy()
fall_data = df[df["Season"] == "Fall"].copy()
winter_data = df[df["Season"] == "Winter"].copy()

# Mapping frekuensi pembelian ke dalam numerik

frequency_ranking = {
    'Weekly': 7,
    'Bi-Weekly': 6,
    'Fortnightly': 6,
    'Monthly': 5,
    'Quarterly': 4,
    'Annually': 3,
    'Every 3 Months': 2
}

# Menambahkan kolom baru untuk frekuensi pembelian dalam numerikal

spring_data["Frequency Ranking"] = df["Frequency of Purchases"].map(frequency_ranking)
summer_data["Frequency Ranking"] = df["Frequency of Purchases"].map(frequency_ranking)
fall_data["Frequency Ranking"] = df["Frequency of Purchases"].map(frequency_ranking)
winter_data["Frequency Ranking"] = df["Frequency of Purchases"].map(frequency_ranking)

# Normalisasi data untuk pembobotan review rating dan frequency rating
def normalize_data(df):
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[['Review Rating', 'Frequency Ranking']] = scaler.fit_transform(df_normalized[['Review Rating', 'Frequency Ranking']])
    return df_normalized
  
#Normalisasi data
spring_data = normalize_data(spring_data)
summer_data = normalize_data(summer_data)
fall_data = normalize_data(fall_data)
winter_data = normalize_data(winter_data)

print(spring_data)

#Preprocessing seasonal data
def preprocessing_seasonal_data(df):
  drop_columns = list(df.columns[0:3]) + list(df.columns[5:9]) + list(df.columns[11:15]) + list(df.columns[16:18])
  df = df.drop(columns=drop_columns)
  
  return df

spring_data = preprocessing_seasonal_data(spring_data)
summer_data = preprocessing_seasonal_data(summer_data)
fall_data = preprocessing_seasonal_data(fall_data)
winter_data = preprocessing_seasonal_data(winter_data)

print(winter_data)

#Cluster total for seasonal data
#Metode elbow untuk mencari total klaster memungkinkan
def find_total_cluster(df):
  score = []
  categorical_columns = list(range(0, 3))
  for k in tqdm(range(2, 7)):
    model = KPrototypes(n_clusters=k)
    model.fit(df, categorical=categorical_columns)
    score.append(model.cost_)
  plt.plot(range(2, 7), score)
  
find_total_cluster(spring_data)
find_total_cluster(summer_data)
find_total_cluster(fall_data)
find_total_cluster(winter_data)

#training data menggunakan KPrototypes sebagai cara klasterisasi untuk kombinasi data kategorikal dan numerik
def train_seasonal_model(df_pred, categorical_columns, name):
  model = KPrototypes(n_clusters=3)
  member = model.fit_predict(df_pred, categorical=categorical_columns)
  df_pred['cluster'] = member
  df_pred.to_csv("..\data\\" + name + ".csv", index=False)

  return model, df_pred


#Copy data asli ke data prediksi
spring_pred = spring_data.copy()
summer_pred = summer_data.copy()
fall_pred = fall_data.copy()
winter_pred = winter_data.copy()


#melatih model dan mendapatkan data klasterisasi
spring_model, spring_clustered_data = train_seasonal_model(spring_pred, list(range(0, 3)), "spring")
summer_model, summer_clustered_data = train_seasonal_model(summer_pred, list(range(0, 3)), "summer")
fallen_model, fall_clustered_data = train_seasonal_model(fall_pred, list(range(0, 3)), "fall")
winter_model, winter_clustered_data = train_seasonal_model(winter_pred, list(range(0, 3)), "winter")

# Preprocessing drop column untuk pemodelan perekomendasian 
drop_columns = [df.columns[0]] + list(df.columns[5:9]) + list(df.columns[11:18])

df = df.drop(columns=drop_columns)

#non seasonal data
score = []
categorical_columns = list(range(1, 5))
for k in tqdm(range(2, 10)):
  model = KPrototypes(n_clusters=k)
  model.fit(df, categorical=categorical_columns)
  score.append(model.cost_)
plt.plot(range(2, 10), score)

#Pelatihan model untuk fitur rekomendasi berdasarkan kriteria pelanggan
model = KPrototypes(n_clusters=4)
member = model.fit_predict(df, categorical=categorical_columns)

df_pred = df.copy()
df_pred["cluster"] = member #memasukkan kelas klaster ke masing-masing data dengan menambahkan kolom klaster
df_pred.to_csv('output_file.csv', index=False)
print(df_pred.head())

#save model

joblib.dump(model, '..\model\model2.joblib')
joblib.dump(spring_model, '..\model\spring_model.joblib')
joblib.dump(summer_model, '..\model\summer_model.joblib')
joblib.dump(winter_model, '..\model\winter_model.joblib')
joblib.dump(fallen_model, r'..\model\fall_model.joblib')


centroids = model.cluster_centroids_

df_centroids = pd.DataFrame(centroids, columns=[
    'Age', 'Review Rating',
    'Gender', 'Item Purchased',
    'Category', 'Season'
])
df_centroids['Age'] = df_centroids['Age'].astype(float).round().astype(int)
print(df_centroids)

#Mendapatkan centroid di data musiman
def centroids_model(model):
  centroids = model.cluster_centroids_
  df_centroids = pd.DataFrame(centroids, columns=[
    'Review Rating', 'Previous Purchases','Frequency Ranking',
    'Item Purchased',
    'Category', 'Season'
  ])
  df_centroids['Previous Purchases'] = df_centroids['Previous Purchases'].astype(float).round().astype(int)
  return df_centroids

spring_centroids = centroids_model(spring_model)
summer_centroids = centroids_model(summer_model)
fall_centroids = centroids_model(fallen_model)
winter_centroids = centroids_model(winter_model)

print(fall_centroids)

# mendapatkan rekomendasi item berdasarkan rating di setiap musim

def get_top_items_by_season_cluster(df_season_clustered, season, cluster_column='cluster', top_n=10):
    max_review_idx = df_season_clustered.groupby([cluster_column, 'Season'])['Review Rating'].idxmax()
    top_items_by_cluster_season = df_season_clustered.loc[max_review_idx, ['cluster', 'Season', 'Item Purchased', 'Review Rating', 'Age']]
    
    # Mengelompokkan berdasarkan klaster dan menampilkan sejumlah N
    top_items_by_season_cluster = (
        top_items_by_cluster_season[top_items_by_cluster_season['Season'] == season]
        .groupby(cluster_column)
        .apply(lambda group: group.nlargest(top_n, 'Review Rating'))
        .reset_index(drop=True)
    )
    
    return top_items_by_season_cluster