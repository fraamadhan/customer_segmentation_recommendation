import streamlit as sl
import pandas as pd
import joblib
from test_season import denormalize_review_rating, get_top_items_by_season_cluster

def load_model():
    model = joblib.load('./model/model1.joblib')
    return model

def create_new_data(age, item_purchase, gender, category, season):
    new_data = pd.DataFrame({
        'Age': [age],
        'Item Purchase': [item_purchase],
        'Gender': [gender],
        'Category': [category],
        'Season': [season]
    })
    return new_data

def recommendation_item(age, gender, category, season):

    # Example usage:
    age_value = age
    item_purchase_value = ''
    gender_value = gender
    category_value = category
    season_value = season

    new_data = create_new_data(age_value, item_purchase_value, gender_value, category_value, season_value)
    
    df_pred = pd.read_csv('./data/output_file.csv')

    model = load_model()

    # Prediksi klaster untuk user baru
    categorical_columns = list(range(1, 5))
    user_cluster = model.predict(new_data, categorical=categorical_columns)

    new_data['cluster'] = user_cluster

    # Rentang umur untuk rekomendasi dari field umur
    age_range = (new_data['Age'].values[0] - 5, new_data['Age'].values[0] + 5)

    # Filter data berdasarkan klaster yang sudah diprediksi dan klaster data baru
    cluster_data = df_pred[df_pred['cluster'] == user_cluster[0]]
    cluster_data = cluster_data[(cluster_data['Age'] >= age_range[0]) & (cluster_data['Age'] <= age_range[1]) &
                                (cluster_data['Gender'] == new_data['Gender'].values[0]) &
                                (cluster_data['Category'] == new_data['Category'].values[0]) &
                                (cluster_data['Season'] == new_data['Season'].values[0])]

    # Mendapatkan top item berdasarkan review rating, item bersifat unik
    top_items = cluster_data.sort_values(by='Review Rating', ascending=False).drop_duplicates('Item Purchased').reset_index(drop=True)
    top_items.index = top_items.index + 1

    # Display the recommended items
    sl.markdown("## Recommended Items:")
    sl.table(top_items[['Item Purchased', 'Category', 'Review Rating', 'Season', 'Age']])
    
winter_clustered_data = pd.read_csv('./data/winter.csv')
summer_clustered_data = pd.read_csv('./data/summer.csv')
fall_clustered_data = pd.read_csv('./data/fall.csv')
winter_clustered_data = pd.read_csv('./data/winter.csv')

winter_clustered_data = denormalize_review_rating(winter_clustered_data)
summer_clustered_data = denormalize_review_rating(summer_clustered_data)
fall_clustered_data = denormalize_review_rating(fall_clustered_data)
winter_clustered_data = denormalize_review_rating(winter_clustered_data)

sl.title("Cicadas Jaya Sandang")
sl.subheader("Selamat datang di Toko Cicadas Jaya Sandang")
sl.caption("Kami menyediakan berbagai jenis kebutuhan Sandang, seperti pakaian, alas kaki, hingga aksesoris")

name = sl.text_input("Nama kamu?", placeholder="Nama")
age = sl.text_input("Umur kamu?", 0)

age = int(str(age))
gender = sl.radio(
    "Jenis Kelamin",
    ["Pria", "Wanita"],
    index=None,
)

if gender == "Pria":
    gender = "Male"
else:
    gender = "Female"

category = sl.selectbox(
    "Kategori apa yang sedang kamu cari?",
    ('Clothing', 'Accessories', 'Footwear', 'Outwear'),
)

season = sl.selectbox(
    "Sedang musim apa di daerahmu?",
    ("Winter", "Summer", "Fall", "winter"),
)

if (sl.button("Dapatkan rekomendasi", type="primary")):
    recommendationCaption = "Hello " + name + " ini ada rekomendasi untukmu: "
    sl.write("Tunggu yaa")
    sl.caption(recommendationCaption)
    recommendation_item(age, gender, category, season)
    
sl.markdown("---")
    
sl.subheader("Rekomendasi item berdasarkan musim")

seasonal_option = sl.selectbox(
    "Pilih musim",
    (("Winter", "Summer", "Fall", "winter")),
)
    
if (seasonal_option == "winter"):
    top_winter_items = get_top_items_by_season_cluster(winter_clustered_data).drop_duplicates('Item Purchased').sort_values('Review Rating', ascending=False).reset_index(drop=True)
    top_winter_items.index += 1
    
    sl.caption("Item rekomendasi untuk winter:")
    sl.table(top_winter_items[['Item Purchased', 'Review Rating']])
    
elif(seasonal_option == "Summer"):
    top_summer_items = get_top_items_by_season_cluster(summer_clustered_data).drop_duplicates('Item Purchased').sort_values('Review Rating', ascending=False).reset_index(drop=True)
    top_summer_items.index += 1
    
    sl.caption("Item rekomendasi untuk summer:")
    sl.table(top_summer_items[['Item Purchased', 'Review Rating']])

elif(seasonal_option == "Fall"):
    top_fall_items = get_top_items_by_season_cluster(fall_clustered_data).drop_duplicates('Item Purchased').sort_values('Review Rating', ascending=False).reset_index(drop=True)
    top_fall_items.index +=1
    
    sl.caption("Item rekomendasi untuk fall:")
    sl.table(top_fall_items[['Item Purchased', 'Review Rating']])

else:
    top_winter_items = get_top_items_by_season_cluster(winter_clustered_data).drop_duplicates('Item Purchased').sort_values('Review Rating', ascending=False).reset_index(drop=True)
    top_winter_items.index += 1
    
    sl.caption("Item rekomendasi untuk winter:")
    sl.table(top_winter_items[['Item Purchased', 'Review Rating']])

