import pandas as pd


class Clustering:
    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        self.model = None

    def data_load(self, new_dataset_path):
        new_data_tmp = pd.read_csv(new_dataset_path)
        self.dataset = pd.concat([self.dataset, new_data_tmp], ignore_index=True)

    def data_visualisation(self):
        pass

    def fit(self, num_clusters):
        pass

    def predict(self):
        pass


if __name__ == "__main__":
    analysis = Clustering('path/to/dataset.csv')

    # Загружаем данные
    analysis.data_load('path/to/dataset2.csv')

    # Визуализируем данные
    analysis.data_visualisation()

    # Обучаем модель
    analysis.fit(num_clusters=3)

    # Предсказываем кластер для новых данных
    new_data = pd.DataFrame({'housing_units': [10, 20, 30],
                             'people_left_homeless': [5, 10, 15],
                             'minors_left_homeless': [2, 4, 6]})
    prediction = analysis.predict(new_data)
    print("Predicted clusters for new data:", prediction)
