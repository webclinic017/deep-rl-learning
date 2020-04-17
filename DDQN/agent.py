import sys
import numpy as np
import keras.backend as K
from keras import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, LSTM, Lambda, Dot, concatenate, TimeDistributed


class Agent:
    """ Agent Class (Network) for DDQN
    """

    def __init__(self, state_dim, action_dim, lr, tau, dueling):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.dueling = dueling
        # Initialize Deep Q-Network
        self.model = self.network(dueling)
        self.model.compile(Adam(), 'mse')
        # Build target Q-Network
        # self.target_model = self.network(dueling)
        # self.target_model.compile(Adam(lr), 'mse')
        # self.target_model.set_weights(self.model.get_weights())

    def huber_loss(self, y_true, y_pred):
        return K.mean(K.sqrt(1 + K.square(y_pred - y_true)) - 1, axis=-1)

    def network(self, dueling):
        """ Build Deep Q-Network
        """
        inp = Input((5, 16))
        x1 = LSTM(1024, dropout=0.1, recurrent_dropout=0.3, return_sequences=True)(inp)
        x1 = LSTM(512, dropout=0.1, recurrent_dropout=0.3, return_sequences=True)(x1)
        x1 = LSTM(512, dropout=0.1, recurrent_dropout=0.3)(x1)
        x1 = Dense(128, activation='relu')(x1)

        inp2 = Input((2,))
        x2 = Dense(128, activation='relu')(inp2)

        output = concatenate([x1, x2])

        output = Dense(128, activation='relu')(output)
        output = Dense(128, activation='relu')(output)

        if dueling:
            # Have the network estimate the Advantage function as an intermediate layer
            output = Dense(self.action_dim + 1, activation='linear')(output)
            output = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                            output_shape=(self.action_dim,))(output)
        else:
            output = Dense(self.action_dim, activation='linear')(output)
        model = Model(inputs=[inp, inp2], output=output)
        model.summary()
        return model

    def transfer_weights(self):
        """ Transfer Weights from Model to Target at rate Tau
        """
        W = self.model.get_weights()
        tgt_W = self.target_model.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
        self.target_model.set_weights(tgt_W)

    def fit(self, inp1, inp2, targ):
        """ Perform one epoch of training
        """
        self.model.fit(x=[inp1, inp2], y=targ, epochs=1, verbose=0)

    def predict(self, inp1, inp2):
        """ Q-Value Prediction
        """
        return self.model.predict(x=[inp1, inp2])

    def target_predict(self, inp1, inp2):
        """ Q-Value Prediction (using target network)
        """
        return self.target_model.predict(x=[inp1, inp2])

    def reshape(self, x):
        # if len(x.shape) < 4 and len(self.state_dim) > 2: return np.expand_dims(x, axis=0)
        # elif len(x.shape) < 3: return np.expand_dims(x, axis=0)
        # else:
        return x

    def save(self, path):
        if (self.dueling):
            path += '_dueling'
        self.model.save_weights(path + '.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
