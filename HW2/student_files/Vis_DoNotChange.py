import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler


def apply_kmeans(dataset_path, n_clusters=4, random_state=42):
    dataset = pd.read_csv(dataset_path)
    le = LabelEncoder()
    for column in ['WaterAccess', 'Sanitation', 'Nutrition']:
        dataset[column] = le.fit_transform(dataset[column])
    features = dataset[['WaterAccess', 'Sanitation', 'Nutrition']]
    features = pd.concat([features, dataset.filter(regex='^Disease_', axis=
        1)], axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10
        )
    cluster_labels = kmeans.fit_predict(features_scaled)
    dataset['Cluster'] = cluster_labels
    return dataset


rural_dataset_path = 'data/rural_health_dataset.csv'
urban_dataset_path = 'data/urban_health_dataset.csv'
rural_dataset = apply_kmeans(rural_dataset_path)
urban_dataset = apply_kmeans(urban_dataset_path)
diseases = ['Respiratory_Infections', 'Diarrheal_Diseases', 'Malaria',
    'HIV_AIDS', 'Parasitic_Infections', 'Malnutrition',
    'Maternal_Mortality', 'Neonatal_Conditions', 'Hypertension',
    'Rheumatic_Heart_Disease', 'Cardiovascular_Diseases', 'Cancers',
    'Mental_Health_Disorders', 'Obesity', 'Respiratory_Diseases',
    'Allergies', 'Autoimmune_Diseases', 'Neurodegenerative_Diseases',
    'Osteoporosis', 'STIs']


def create_combined_heatmap(rural_dataset, urban_dataset, diseases):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, dataset, title in zip(axes, [rural_dataset, urban_dataset], [
        'Harborview', 'Greenfield']):
        sns.heatmap(dataset.groupby('Cluster')[diseases].mean(), cmap=
            'YlGnBu', annot=True, fmt='.2f', ax=ax)
        ax.set_title(f'Heatmap of Disease Prevalence by Cluster in {title}')
        ax.set_xlabel('Disease')
        ax.set_ylabel('Cluster')
        ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.show()


create_combined_heatmap(rural_dataset, urban_dataset, diseases)
