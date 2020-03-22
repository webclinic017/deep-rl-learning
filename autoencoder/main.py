import numpy as np
import pandas as pd
from keras import Sequential, Model, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, BatchNormalization, LSTM, RepeatVector, TimeDistributed, Flatten
from keras.utils import plot_model
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns


class Autoencoder:
    def __init__(self):
        self.train_data, self.normalize_price, self.raw_price, self.moving_avg = self.get_data('../data/15minute.csv')
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
        model.add(Dense(128, input_dim=1, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
        return model

    def create_dataset(self, dataset, look_back=20):
        dataX, dataY = [], []
        num_positive = 0
        num_negative = 0
        for i in range(len(dataset) - look_back - 2):
            a = dataset[['CCI', 'k_fast', 'd_fast']].iloc[i:i+look_back].values
            a = a.flatten()
            close_data = dataset[['Open', 'Close']].iloc[i+look_back-1].values
            diff = close_data[1] - close_data[0]

            if diff > 10:
                num_positive += 1
                dataX.append(a)
                dataY.append(1)

            if diff < -10:
                num_negative += 1
                dataX.append(a)
                dataY.append(0)

        return np.array(dataX), np.array(dataY)

    def train_start(self):
        col = ['CCI', 'k_fast', 'd_fast']
        train_data = self.train[col]
        self.train['signal'] = 0
        self.train.loc[(self.train.Open < self.train.Close), 'signal'] = 1
        # self.train.loc[(self.train.Close < self.train.Open), 'signal'] = 0
        x_train, y_train = self.create_dataset(self.train)
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        normalized = scaler.transform(x_train)
        # define model
        input_img = Input(shape=(60,))
        encoded = Dense(128, activation='relu')(input_img)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(1, activation='relu')(encoded)

        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(60, activation='sigmoid')(decoded)
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.summary()

        autoencoder.fit(normalized, normalized,
                        epochs=100,
                        batch_size=32,
                        shuffle=True,
                        verbose=0
                        )
        # connect the encoder LSTM as the output layer
        model = Model(inputs=autoencoder.inputs, outputs=autoencoder.layers[0].output)
        # plot_model(model, show_shapes=True, to_file='lstm_encoder.png')
        # get the feature vector for the input sequence
        encoder = Model(autoencoder.inputs, encoded)
        encoder.save_weights('model_weight.h5')

    def evaluate(self):
        col = ['CCI', 'k_fast', 'd_fast']
        train_data = self.train[col]
        self.train['signal'] = 0
        self.train.loc[(self.train.Open < self.train.Close), 'signal'] = 1
        # self.train.loc[(self.train.Close < self.train.Open), 'signal'] = 0
        x_train, y_train = self.create_dataset(self.train)
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        normalized = scaler.transform(x_train)
        input_img = Input(shape=(60,))
        encoded = Dense(128, activation='relu')(input_img)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(1, activation='relu')(encoded)

        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(60, activation='sigmoid')(decoded)
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        encoder = Model(autoencoder.inputs, encoded)
        encoder.load_weights('model_weight.h5')
        encoded_imgs = encoder.predict(normalized)
        # encoder_scaler = MinMaxScaler(feature_range=(min(encoded_imgs)[0], max(encoded_imgs)[0]))
        # encoder_scaler.fit_transform(y_train.reshape(-1, 1))
        # matplotlib histogram
        # seaborn histogram
        # colors = ['#E69F00', '#56B4E9']
        # names = ['Encoder', 'Price']
        # plt.hist([encoded_imgs.flatten(), encoder_scaler.transform(y_train.reshape(1, -1)).flatten()], bins=int(180/15), color=colors, label=names, stacked=True,)
        # # Add labels
        # plt.legend()
        # plt.title('Histogram of Arrival Delays')
        # plt.xlabel('Delay (min)')
        # plt.ylabel('Flights')
        # plt.show()

        self.re_train(encoded_imgs, y_train)

    def re_train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123)

        model = KerasClassifier(build_fn=self.baseline_model, epochs=10000, batch_size=5, verbose=0)
        # kfold = KFold(n_splits=2)
        # results = cross_val_score(model, X_train, y_train, cv=kfold)
        # print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

        model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[es, mc], verbose=1)
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

    def norm_list(self, list_needed):
        """Scale feature to 0-1"""
        min_x = min(list_needed)
        max_x = max(list_needed)
        if not min_x or not max_x:
            return list_needed

        return_data = list(map(lambda x: (x - min_x) / (max_x - min_x), list_needed))
        return return_data

if __name__ == '__main__':
    autoencode = Autoencoder()
    autoencode.train_start()
    autoencode.evaluate()
