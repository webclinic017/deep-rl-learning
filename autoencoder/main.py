import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, BatchNormalization, LSTM, RepeatVector, TimeDistributed
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from matplotlib import pyplot


class Autoencoder:
    def __init__(self):
        self.train_data, self.normalize_price, self.raw_price, self.moving_avg = self.get_data('../data/1hour.csv')
        self.cci = self.calculate_cci(self.train_data, 25)
        self.train = self.macd(self.cci)
        self.train = self.stochastics(self.train)
        self.train = self.train.dropna()
        self.budget = 0

    @staticmethod
    def normalize(raw_list):
        """Scale feature to 0-1"""
        min_x = min(raw_list)
        max_x = max(raw_list)
        if not min_x or not max_x:
            return raw_list

        return_data = list(map(lambda x: (x - min_x) / (max_x - min_x), raw_list))
        return return_data

    def get_data(self, csv_path):
        df = pd.read_csv(csv_path, sep=',')
        normalize_price = self.normalize(df.Close.values)
        raw_price = df.Close.values
        moving_avg = df.Close.rolling(window=9).mean().dropna().values
        return df, normalize_price, raw_price, moving_avg

    @staticmethod
    def macd(data_frame):
        data = data_frame.copy()
        ma = data_frame.Close.rolling(window=14).mean().fillna(0)
        exp1 = data.Close.ewm(span=12, adjust=False).mean()
        exp2 = data.Close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2

        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return_data = data_frame.join(pd.Series(signal, name='Signal'))
        return_data = return_data.join(pd.Series(histogram, name='Histogram'))
        return_data = return_data.join(pd.Series(macd, name='MACD'))
        return_data = return_data.join(pd.Series(ma, name='MA'))
        return return_data

    @staticmethod
    def calculate_cci(dataraw, ndays):
        """Commodity Channel Index"""
        tp = (dataraw['High'] + dataraw['Low'] + dataraw['Close']) / 3
        raw_cci = pd.Series((tp - tp.rolling(ndays).mean()) / (0.015 * tp.rolling(ndays).std()), name='CCI')
        return_data = dataraw.join(raw_cci)
        return return_data

    @staticmethod
    def stochastics(data_frame, k=14, d=3):
        """
        Fast stochastic calculation
        %K = (Current Close - Lowest Low)/
        (Highest High - Lowest Low) * 100
        %D = 3-day SMA of %K

        Slow stochastic calculation
        %K = %D of fast stochastic
        %D = 3-day SMA of %K

        When %K crosses above %D, buy signal
        When the %K crosses below %D, sell signal
        """
        df = data_frame.copy()

        # Set minimum low and maximum high of the k stoch
        low_min = df.Low.rolling(window=k).min()
        high_max = df.High.rolling(window=k).max()

        # Fast Stochastic
        df['k_fast'] = 100 * (df.Close - low_min) / (high_max - low_min)
        df['d_fast'] = df['k_fast'].rolling(window=d).mean()

        # Slow Stochastic
        df['k_slow'] = df["d_fast"]
        df['d_slow'] = df['k_slow'].rolling(window=d).mean()
        return df

    # define base model
    def baseline_model(self):
        # create model
        model = Sequential()
        model.add(Dense(512, input_dim=3, activation='relu', kernel_initializer='uniform'))
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu', kernel_initializer='uniform'))
        model.add(Dense(1, activation='softmax', kernel_initializer='uniform'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
        return model

    def lstm_autoencoder(self, n_in=10):
        model = Sequential()
        model.add(LSTM(100, activation='relu', input_shape=(n_in, 1)))
        model.add(RepeatVector(n_in))
        model.add(LSTM(100, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(1)))
        model.compile(optimizer='adam', loss='mse')
        return model

    def create_dataset(self, dataset, look_back=10):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[['CCI', 'k_fast', 'd_fast']].iloc[i:i+look_back].values
            dataX.append(a)
        return np.array(dataX), np.array(dataX)

    def train_start(self):
        col = ['CCI', 'k_fast', 'd_fast']
        train_data = self.train[col]
        self.train['signal'] = 0
        self.train.loc[(self.train.Open < self.train.Close), 'signal'] = 1
        # self.train.loc[(self.train.Close < self.train.Open), 'signal'] = 0
        x_train, y_train = self.create_dataset(self.train)

        # y = self.train.signal
        # X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.2)
        # lm = linear_model.LinearRegression(normalize=True)
        # model = lm.fit(X_train, y_train)
        # predictions = lm.predict(X_test)
        # for correct, prediction in zip(y, predictions):
        #     print(correct, prediction)

        # evaluate model
        # standardscaler = StandardScaler()
        # standardscaler.fit_transform(x_train)

        model = KerasClassifier(build_fn=self.baseline_model, epochs=10000, batch_size=5, verbose=0)
        # kfold = KFold(n_splits=2)
        # results = cross_val_score(model, X_train, y_train, cv=kfold)
        # print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

        model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[es, mc])
        y_pred = model.predict(X_test)
        y_true = y_test
        print(accuracy_score(y_true, y_pred))
        lr_probs = model.predict_proba(X_test)
        lr_probs = lr_probs[:, 1]
        # calculate scores
        ns_probs = [0 for _ in range(len(y_test))]
        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, lr_probs)
        # summarize scores
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('Logistic: ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
        # plot the roc curve for the model
        pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()
        #
        # # predict probabilities
        yhat = model.predict_proba(X_test)
        # retrieve just the probabilities for the positive class
        pos_probs = yhat[:, 1]
        no_skill = len(y[y == 1]) / len(y)
        pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        # calculate model precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, pos_probs)
        # plot the model precision-recall curve
        pyplot.plot(recall, precision, marker='.', label='Logistic')
        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()

        print("Done")

        # print(Counter(yhat))
        # create a histogram of the predicted probabilities
        pyplot.hist(pos_probs, bins=100)
        pyplot.show()

        # print(y_pred)

        # estimator = KerasRegressor(build_fn=self.baseline_model, epochs=100, batch_size=5, verbose=1)
        # kfold = KFold(n_splits=10)
        # results = cross_val_score(estimator, X_train, y_train, cv=kfold)
        # print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))


if __name__ == '__main__':
    autoencode = Autoencoder()
    autoencode.train_start()
