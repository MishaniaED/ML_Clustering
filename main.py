import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


class Clustering:
    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        self.model = None

    def data_load(self, new_dataset_path):
        new_data_tmp = pd.read_csv(new_dataset_path)
        self.dataset = pd.concat([self.dataset, new_data_tmp], ignore_index=True)

    def process_data(self):
        self.dataset["date_of_demolition"] = pd.to_datetime(self.dataset["date_of_demolition"])
        self.dataset.dropna(inplace=True)

    def display_dataset_properties(self):
        print(
            f"СВОЙСТВА ДАТАСЕТА\n\nКоличество строк: {self.dataset.shape[0]}\nКоличество столбцов: "
            f"{self.dataset.shape[1]}\n")
        self.dataset.info()

    def display_dataset_content(self):
        print("СОДЕРЖАНИЕ ДАТАСЕТА\n")
        self.dataset.head()

    def display_missing_values(self):
        print("КОЛИЧЕСТВО NA В ДАТАСЕТЕ ПО СТОЛБЦАМ\n")
        self.dataset["date_of_demolition"] = pd.to_datetime(self.dataset["date_of_demolition"])
        self.dataset.isna().sum()
        self.dataset.dropna(inplace=True)
        self.dataset.describe().T

    def visualize_pie_chart(self, data, labels, title):
        explode = (0.1, 0.1, 0.1)
        plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=140, explode=explode)
        plt.title(title)
        plt.axis('equal')
        plt.show()

    def visualize_bar_chart(self, data, xlabel, ylabel, title):
        plt.figure(figsize=(10, 6))
        data.plot(kind='bar', color='skyblue')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.show()

    def display_age_distribution_pie_chart(self):
        total_people_left_homeless = self.dataset['people_left_homeless'].sum()
        minors_left_homeless = self.dataset['minors_left_homeless'].sum()
        percentage_minors = (minors_left_homeless / total_people_left_homeless) * 100
        percentage_adults = 100 - percentage_minors
        labels = ['Minors', 'Adults']
        sizes = [percentage_minors, percentage_adults]
        colors = ['lightblue', 'lightcoral']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, explode=(0.05, 0.05))
        plt.title('Соотношение возраста людей, оставшихся без места проживания')
        plt.axis('equal')
        plt.show()

    def display_trend_of_demolitions(self):
        self.dataset['date_of_demolition'] = pd.to_datetime(self.dataset['date_of_demolition'])
        self.dataset.set_index('date_of_demolition', inplace=True)
        data_resampled = self.dataset.resample('M').count()
        plt.figure(figsize=(12, 6))
        plt.plot(data_resampled.index, data_resampled['locality'], marker='o')
        plt.title('Тенденция разрушений по годам')
        plt.xlabel('Год')
        plt.ylabel('Количество разрушенных построек')
        plt.grid()
        plt.show()

    def display_top_localities(self):
        locality_counts = self.dataset['locality'].value_counts()
        print("СПИСОК РАЙОНОВ С НАИБОЛЬШИМ КОЛИЧЕСТВОМ РАЗРУШЕНИЙ ПО УБЫВАНИЮ\n")
        print(locality_counts)
        top_10_localities = locality_counts.head(10).index.tolist()
        filtered_locality_counts = locality_counts[top_10_localities]
        plt.figure(figsize=(12, 6))
        filtered_locality_counts.plot(kind='bar')
        plt.title('Топ 10 районов с наибольшим количеством разрушений')
        plt.xlabel('Район')
        plt.ylabel('Количество разрушенных построек')
        plt.xticks(rotation=45)
        plt.show()

    def display_district_demolitions(self):
        district_counts = self.dataset['district'].value_counts()
        print("СПИСОК ГОРОДОВ С НАИБОЛЬШИМ КОЛИЧЕСТВОМ РАЗРУШЕНИЙ ПО УБЫВАНИЮ\n")
        print(district_counts)
        plt.figure(figsize=(12, 6))
        district_counts.plot(kind='bar')
        plt.title('Распределение разрушений по городам')
        plt.xlabel('Город')
        plt.ylabel('Количество разрушенных построек')
        plt.xticks(rotation=45)
        plt.savefig("3.png")
        plt.show()

    def display_area_demolitions(self):
        area_counts = self.dataset['area'].value_counts()
        print("КОЛИЧЕСТВО РАЗРУШЕННЫХ ПОСТРОЕК ПО РЕГИОНАМ\n")
        print(area_counts)
        plt.figure(figsize=(12, 6))
        area_counts.plot(kind='bar')
        plt.title('Распределение разрушений по регионам')
        plt.xlabel('Регион')
        plt.ylabel('Количество разрушенных построек')
        plt.xticks(rotation=45)
        plt.savefig("4.png")
        plt.show()

    def display_relationship_housing_units_people(self):
        relevant_data = self.dataset[['housing_units', 'people_left_homeless', 'minors_left_homeless']]
        plt.figure(figsize=(10, 6))
        plt.scatter(relevant_data['housing_units'], relevant_data['people_left_homeless'], color='b', label='Все люди')
        plt.scatter(relevant_data['housing_units'], relevant_data['minors_left_homeless'], color='r',
                    label='Несовершеннолетние')
        plt.title('Соотношение возраста людей, оставшихся без места жительства, к количеству разрушенных построек')
        plt.xlabel('Количество разрушенных построек')
        plt.ylabel('Количество разрушенных построек')
        plt.legend()
        plt.savefig("5.png")
        plt.show()

    def display_correlation_matrix(self):
        correlation = self.dataset[['housing_units', 'people_left_homeless', 'minors_left_homeless']].corr()
        print("\nКОЭФФИЦИЕНТ КОРЕЛЛЯЦИИ\n")
        print(correlation)

    def display_reason_demolition(self):
        reason_counts = self.dataset['reason_for_demolition'].value_counts()
        plt.figure(figsize=(10, 6))
        reason_counts.plot(kind='bar', color='skyblue')
        plt.title('Причины разрушения построект')
        plt.xlabel('Причина')
        plt.ylabel('Частота причины')
        plt.xticks(rotation=45)
        plt.savefig("6.png", dpi=300, bbox_inches='tight')
        plt.show()
        reason_percentage = self.dataset['reason_for_demolition'].value_counts(normalize=True) * 100
        print("\nСООТНОШЕНИЕ ПРИЧИН РАЗРУШЕНИЯ ПОСТРОЕК:\n")
        print(reason_percentage)

    def display_yearly_trend(self):
        self.dataset.index = pd.to_datetime(self.dataset.index)
        yearly_data = self.dataset.resample('Y').count()
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_data.index, yearly_data['locality'], marker='o')
        plt.title('Тенденция разрушений по годам')
        plt.xlabel('Год')
        plt.ylabel('Количество разрушений')
        plt.grid()
        plt.savefig("7.png")
        plt.show()

    def display_seasonal_decomposition(self):
        result = seasonal_decompose(self.dataset['housing_units'], model='additive', period=1)
        result.plot()
        plt.savefig("7-1.png")
        plt.show()

    def display_autocorrelation_function(self):
        plot_acf(self.dataset['housing_units'], lags=30)
        plt.savefig("7-2.png")
        plt.show()

    def display_impact_analysis(self):
        impact_analysis = self.dataset.groupby('reason_for_demolition').agg(
            {'people_left_homeless': 'sum', 'minors_left_homeless': 'sum'})
        impact_analysis.plot(kind='bar', figsize=(10, 6))
        plt.title('Соотношение возрастных групп людей, оставшихся без места жительства по причинам разрушения построек')
        plt.xlabel('Причина разрушения постройки')
        plt.ylabel('Количество пострадавших')
        plt.xticks(rotation=45)
        plt.savefig("8.png", dpi=300, bbox_inches='tight')
        plt.show()

    def display_structure_demolitions(self):
        structure_counts = self.dataset.groupby('type_of_sturcture')['housing_units'].count()
        print("Количество разрушенных построек в зависимости от их типа\n")
        print(structure_counts)
        structure_counts.plot(kind='bar', figsize=(10, 6), color='skyblue')
        plt.title('Количество разрушенных построек в зависимости от их типа')
        plt.xlabel('Тип постройки')
        plt.ylabel('Количество разрушенных построек')
        plt.xticks(rotation=45)
        plt.savefig("9.png", dpi=300, bbox_inches='tight')
        plt.show()
        structure_percentage = structure_counts / structure_counts.sum() * 100
        print("\nСООТНОШЕНИЕ РАЗРУШЕННЫХ ПОСТРОЕК В ЗАВИСИМОСТИ ОТ ИХ ТИПА\n")
        print(structure_percentage)

    def display_minors_left_homeless_distribution(self):
        sns.displot(self.dataset[self.dataset['minors_left_homeless'] != 0]['minors_left_homeless'], bins=20, kde=True,
                    height=8,
                    aspect=2)
        plt.show()

    def data_visualisation(self):
        self.display_dataset_properties()
        self.display_dataset_content()
        self.display_missing_values()

        self.visualize_pie_chart(self.dataset['area'].value_counts(), self.dataset['area'].value_counts().index,
                                 'Соотношение регионов в датасете')

        self.visualize_bar_chart(self.dataset['reason_for_demolition'].value_counts(), 'Причина разрушения',
                                 'Частота причины', 'Соотношение причин разрушений построек')

        self.display_age_distribution_pie_chart()
        self.display_trend_of_demolitions()
        self.display_top_localities()
        self.display_district_demolitions()
        self.display_area_demolitions()

        self.display_relationship_housing_units_people()
        self.display_correlation_matrix()
        self.display_reason_demolition()
        self.display_yearly_trend()

        self.display_seasonal_decomposition()
        self.display_autocorrelation_function()
        self.display_impact_analysis()

        self.display_structure_demolitions()

        self.display_minors_left_homeless_distribution()

    def preprocess_data_for_clustering(self):
        column_name = 'locality_encoded_fr'
        columns_to_check = ['date_of_demolition', 'locality', 'people_left_homeless']
        rows_with_nan = self.dataset[self.dataset[columns_to_check].isna().any(axis=1)]

        self.dataset = self.dataset.dropna(subset=columns_to_check)

        data = self.dataset
        data['date_of_demolition'] = pd.to_datetime(data['date_of_demolition'])
        data['date_of_demolition'] = (
                data['date_of_demolition'].astype('datetime64[ns]').astype('int64') // 10 ** 9).astype(int)

        plt.figure(figsize=(20, 12))
        plt.scatter(data['date_of_demolition'], data['district'], marker='o', s=30)
        plt.title('Clustered Data')
        plt.xlabel('Date of Demolition')
        plt.yticks(rotation=45)
        plt.ylabel('People Left Homeless')
        plt.show()

        locality_counts = data['locality'].value_counts().to_dict()
        data['locality_encoded_fr'] = data['locality'].map(locality_counts)

        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(
            data[['date_of_demolition', 'locality_encoded_fr', 'people_left_homeless']])

        return data, normalized_data

    def apply_dbscan(self, normalized_data, eps_value, min_samples_value):
        dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
        dbscan.fit(normalized_data)

        # Получаем метки кластеров
        cluster_labels = dbscan.labels_

        # Добавляем метки кластеров в DataFrame
        self.dataset['cluster_dbscan'] = cluster_labels

    def apply_kmeans(self, normalized_data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.dataset['cluster_Kmeans'] = kmeans.fit_predict(normalized_data)

    def visualize_clusters(self, cluster_column):
        plt.figure(figsize=(20, 12))
        plt.scatter(self.dataset['date_of_demolition'], self.dataset['district'], c=self.dataset[cluster_column],
                    cmap='viridis', marker='o', s=30)
        plt.title('Clustered Data')
        plt.xlabel('Date of Demolition')
        plt.ylabel('District')
        plt.show()

    def elbow_method_for_optimal_k(self, normalized_data):
        inertia = []

        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(normalized_data)
            inertia.append(kmeans.inertia_)

        # Построение графика метода локтя
        plt.plot(range(1, 11), inertia, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.show()

    def clustering(self, eps_value=0.5, min_samples_value=5, n_clusters=8):
        data, normalized_data = self.preprocess_data_for_clustering()

        self.apply_dbscan(normalized_data, eps_value, min_samples_value)
        self.apply_kmeans(normalized_data, n_clusters)

        self.visualize_clusters('cluster_Kmeans')
        self.visualize_clusters('cluster_dbscan')

        self.elbow_method_for_optimal_k(normalized_data)


if __name__ == "__main__":
    analysis = Clustering('demolitions_pse_isr_conflict_2004-01_to_2023-08.csv')

    # Загружаем данные
    # analysis.data_load('path/to/dataset2.csv')

    # Визуализируем данные
    analysis.data_visualisation()

    analysis.clustering()

