import pandas as pd

def denormalize_review_rating(df, original_min=1, original_max=5):
    df_denormalized = df.copy()
    df_denormalized['Review Rating'] = round(df_denormalized['Review Rating'] * (original_max - original_min) + original_min, 2)
    return df_denormalized

def get_top_items_by_season_cluster(df_season_clustered, cluster_column='cluster', top_n=10):
    max_review_freq_idx = df_season_clustered.groupby([cluster_column, 'Season'])[['Review Rating', 'Frequency Ranking']].idxmax()
    top_items_by_cluster_season = df_season_clustered.loc[max_review_freq_idx[['Review Rating', 'Frequency Ranking']].values.flatten(), ['cluster', 'Season', 'Item Purchased', 'Review Rating', 'Frequency Ranking']]

    # Filter item dengan review rating > 3
    top_items_by_cluster_season = top_items_by_cluster_season[top_items_by_cluster_season['Review Rating'] > 3]

    # Grup berdasarkan klaster dan jumlah n
    top_items_by_season_cluster = (
        top_items_by_cluster_season
        .groupby(cluster_column, as_index=False)
        .apply(lambda group: group.nlargest(top_n, columns=['Review Rating', 'Frequency Ranking']))
        .reset_index(drop=True)
    )

    return top_items_by_season_cluster

spring_clustered_data = pd.read_csv('./data/spring.csv')
summer_clustered_data = pd.read_csv('./data/summer.csv')
fall_clustered_data = pd.read_csv('./data/fall.csv')
winter_clustered_data = pd.read_csv('./data/winter.csv')


spring_clustered_data = denormalize_review_rating(spring_clustered_data)
summer_clustered_data = denormalize_review_rating(summer_clustered_data)
fall_clustered_data = denormalize_review_rating(fall_clustered_data)
winter_clustered_data = denormalize_review_rating(winter_clustered_data)
