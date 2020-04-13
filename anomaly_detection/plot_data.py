import ta
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt


class TradingEnv:
    def __init__(self, dataset=None):
        self.train_data, self.prices = self.get_data(dataset)

    def get_data(self, csv_path):
        df = pd.read_csv(csv_path, sep=',')
        df = df.drop(columns=['no'], axis=1)
        # df = df.drop(columns=['open_time', 'close_time', 'ignore'], axis=1)
        ta.add_volume_ta(df, close='close', high='high', low='low', volume='volume', fillna=True)
        df = df.dropna()
        raw_price = df.close.astype('float64').values
        data = df.copy()
        scaler_filename = 'scaler.pkl'
        pca_filename = 'pca.pkl'
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data)
        data = scaler.transform(data)

        # PCA
        # pca = decomposition.PCA(n_components=self.nb_features, random_state=123)
        # pca.fit(data)
        # joblib.dump(pca, pca_filename)
        # data = pca.transform(data)
        # print(pca.explained_variance_ratio_)
        # print(pca.singular_values_)
        columns = ['open', 'high', 'low', 'close',
                   'volume', 'volume_adi', 'volume_obv', 'volume_cmf',
                   'volume_fi', 'momentum_mfi', 'volume_em', 'volume_sma_em',
                   'volume_vpt', 'volume_nvi']
        fig, ax = plt.subplots(nrows=7, ncols=2)
        curr_index = 0
        for row_index, row in enumerate(ax):
            for col_index, col in enumerate(row):
                col.plot(data[:, curr_index], label=columns[curr_index])
                col.legend()
                curr_index += 1

        plt.hist(data[:, -2], bins=50)
        plt.show()

        return df, raw_price

    def isolation_forests(self, outliers_fraction=0.001):
        fig, ax = plt.subplots(figsize=(10, 6))
        # for x in range(24, len(self.train_data)):
        plt.cla()
        data = self.train_data.copy()
        scaler = StandardScaler()
        scaler.fit(data)
        np_scaled = scaler.transform(data)
        scaler_data = pd.DataFrame(np_scaled)

        # train isolation forest
        model = IsolationForest(contamination=outliers_fraction)
        model.fit(scaler_data)
        predict = model.predict(scaler_data)
        data = data.reset_index()
        data['anomaly2'] = pd.Series(predict)

        a = data.loc[data['anomaly2'] == -1, ['close', 'index']]  # anomaly
        # visualization

        ax.plot(data['index'], data['close'], color='blue', label='Normal')
        ax.scatter(a['index'], a['close'], color='red', label='Anomaly')
        plt.show()


if __name__ == '__main__':
    env = TradingEnv(dataset="../data/tct_1d.csv")
    env.isolation_forests()
