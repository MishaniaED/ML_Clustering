import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf


class Clustering:
    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        self.model = None

    def data_load(self, new_dataset_path):
        new_data_tmp = pd.read_csv(new_dataset_path)
        self.dataset = pd.concat([self.dataset, new_data_tmp], ignore_index=True)

    def data_visualisation(self):
        print(
            f"СВОЙСТВА ДАТАСЕТА\n\nКоличество строк: {self.dataset.shape[0]}\nКоличество столбцов: "
            f"{self.dataset.shape[1]}\n")
        self.dataset.info()

        print("СОДЕРЖАНИЕ ДАТАСЕТА\n")
        self.dataset.head()

        print("КОЛИЧЕСТВО NA В ДАТАСЕТЕ ПО СТОЛБЦАМ\n")
        self.dataset["date_of_demolition"] = pd.to_datetime(self.dataset["date_of_demolition"])
        data_original = self.dataset
        self.dataset.isna().sum()

        self.dataset.dropna(inplace=True)
        self.dataset.describe().T

        value_counts = self.dataset['area'].value_counts()
        explode = (0.1, 0.1, 0.1)
        plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140, explode=explode)
        plt.title('Соотношение регионов в датасете')
        plt.axis('equal');

        value_counts = self.dataset['reason_for_demolition'].value_counts()
        explode = (0.05, 0.05, 0.05)
        plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140, explode=explode)
        plt.title('Соотношение причин разрушений построек')
        plt.axis('equal');

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

        self.dataset['date_of_demolition'] = pd.to_datetime(self.dataset['date_of_demolition'])
        self.dataset.set_index('date_of_demolition', inplace=True)
        data_resampled = self.dataset.resample('M').count()
        plt.figure(figsize=(12, 6))
        plt.plot(data_resampled.index, data_resampled['locality'], marker='o')
        plt.title('Тендеция разрушений по годам')
        plt.xlabel('Год')
        plt.ylabel('Количество разрушенных построек')
        plt.grid()
        plt.show()

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
        correlation = relevant_data.corr()
        print("\nКОЭФФИЦИЕНТ КОРЕЛЛЯЦИИ\n")
        print(correlation)

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

        self.dataset.index = pd.to_datetime(self.dataset.index)
        yearly_data = self.dataset.resample('Y').count()
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_data.index, yearly_data['locality'], marker='o')
        plt.title('Тенденция разрущений по годам')
        plt.xlabel('Год')
        plt.ylabel('Количество разрушений')
        plt.grid()
        plt.savefig("7.png")
        plt.show()

        result = seasonal_decompose(self.dataset['housing_units'], model='additive', period=1)
        result.plot()
        plt.savefig("7-1.png")
        plt.show()

        plot_acf(self.dataset['housing_units'], lags=30)
        plt.savefig("7-2.png")
        plt.show()

        impact_analysis = self.dataset.groupby('reason_for_demolition').agg(
            {'people_left_homeless': 'sum', 'minors_left_homeless': 'sum'})
        impact_analysis.plot(kind='bar', figsize=(10, 6))
        plt.title('Соотношение возрастных групп людей, оставшихся без места жительства по причинам разрушения построек')
        plt.xlabel('Причина разрушения постройки')
        plt.ylabel('Количество пострадавших')
        plt.xticks(rotation=45)
        plt.savefig("8.png", dpi=300, bbox_inches='tight')
        plt.show()

        print("ЧАСТОТА ПРИЧИН РАЗРУШЕНИЯ ПОСТРОЕК В ЗАВИСИМОСТИ ОТ ВОЗРАСТНОЙ ГРУППЫ:\n")
        print(impact_analysis)

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

        data_original["date_of_demolition"] = pd.to_datetime(data_original['date_of_demolition'], errors='coerce')
        grouped_demolition = self.dataset.groupby(
            [data_original["date_of_demolition"].dt.year, "type_of_sturcture"]).size()
        df_demolition = grouped_demolition.reset_index().rename(columns={0: 'count'})
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_demolition, x='date_of_demolition', y='count', hue='type_of_sturcture')
        plt.title('Количество и типы разрушенных построек по годам')
        plt.xlabel('Год')
        plt.ylabel('Количество разрушенных построек')
        plt.legend(title='Тип постройки')
        plt.show()

        sns.displot(self.dataset[self.dataset['minors_left_homeless'] != 0]['minors_left_homeless'], bins=20, kde=True,
                    height=8,
                    aspect=2)
        plt.show()

    def fit(self, num_clusters):
        pass

    def predict(self):
        pass


if __name__ == "__main__":
    analysis = Clustering('demolitions_pse_isr_conflict_2004-01_to_2023-08.csv')

    # Загружаем данные
    # analysis.data_load('path/to/dataset2.csv')

    # Визуализируем данные
    analysis.data_visualisation()

    # # Обучаем модель
    # analysis.fit(num_clusters=3)
    #
    # # Предсказываем кластер для новых данных
    # new_data = pd.DataFrame({'housing_units': [10, 20, 30],
    #                          'people_left_homeless': [5, 10, 15],
    #                          'minors_left_homeless': [2, 4, 6]})
    # prediction = analysis.predict(new_data)
    # print("Predicted clusters for new data:", prediction)
