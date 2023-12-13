import streamlit as sl
import pandas as pd
import joblib

def load_model():
    model = joblib.load('model1.joblib')
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
    
    df_pred = pd.read_csv('output_file.csv')

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
    top_items = cluster_data.sort_values(by='Review Rating', ascending=False).drop_duplicates('Item Purchased')

    # Display the recommended items
    sl.markdown("## Recommended Items:")
    sl.table(top_items[['Item Purchased', 'Category', 'Review Rating', 'Season', 'Age']])

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
    ("Winter", "Summer", "Fall", "Spring"),
)

if (sl.button("Dapatkan rekomendasi", type="primary")):
    recommendationCaption = "Hello " + name + " ini ada rekomendasi untukmu: "
    sl.write("Tunggu yaa")
    sl.caption(recommendationCaption)
    recommendation_item(age, gender, category, season)
    

