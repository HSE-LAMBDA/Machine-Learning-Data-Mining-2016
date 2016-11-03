import numpy as np

import theano
import theano.tensor as T

from lasagne import *

from net import Net

class CAE(Net):
    def __init__(self, n_codes=25):
        self.n_codes = n_codes
        X_batch = T.ftensor4(name='images')

        in_l = layers.InputLayer(shape=(None, 1, 28, 28), input_var=X_batch)
        conv1 = layers.Conv2DLayer(
            in_l,
            num_filters=8,
            filter_size=(3, 3),
            nonlinearity=nonlinearities.elu
        )
        pool1 = layers.Pool2DLayer(
            conv1,
            pool_size=(2, 2)
        )
        
        conv2 = layers.Conv2DLayer(
            pool1,
            num_filters=16,
            filter_size=(2, 2),
            nonlinearity=nonlinearities.elu
        )
        pool2 = layers.Pool2DLayer(
            conv2,
            pool_size=(2, 2)
        )
        
        conv3 = layers.Conv2DLayer(
            pool2,
            num_filters=32,
            filter_size=(3, 3),
            nonlinearity=nonlinearities.elu
        )
        pool3 = layers.Pool2DLayer(
            conv3,
            pool_size=(2, 2)
        )
        
        flatten = layers.FlattenLayer(pool3)
        
        dense1 = layers.DenseLayer(
            flatten,
            num_units = n_codes,
            nonlinearity=nonlinearities.sigmoid
        )
        
        code = layers.get_output(dense1)
        
        dense2 = layers.DenseLayer(
            dense1,
            num_units=128,
            nonlinearity=nonlinearities.sigmoid
        )
        
        dedense = layers.ReshapeLayer(
            dense2,
            shape=(-1, 32, 2, 2)
        )
        
        upscale1 = layers.Upscale2DLayer(
            dedense,
            scale_factor=2
        )
        
        deconv1 = layers.Deconv2DLayer(
            upscale1,
            num_filters=16,
            filter_size=(3, 3),
            crop='valid',
            nonlinearity=nonlinearities.elu
        )
        
        upscale2 = layers.Upscale2DLayer(
            deconv1,
            scale_factor=2
        )
        
        deconv2 = layers.Deconv2DLayer(
            upscale2,
            num_filters=8,
            filter_size=(2, 2),
            crop='valid',
            nonlinearity=nonlinearities.elu
        )
        
        upscale3 = layers.Upscale2DLayer(
            deconv2,
            scale_factor=2
        )
        
        deconv3 = layers.Deconv2DLayer(
            upscale3,
            num_filters = 1,
            filter_size=(3, 3),
            crop='valid',
            nonlinearity=nonlinearities.sigmoid
        )
        
        self.net = deconv3
        
        X_reconstructed = layers.get_output(self.net)
        
        log_loss = -T.mean(X_batch * T.log(X_reconstructed) + (1 - X_batch) * T.log(1 - X_reconstructed))
        mse_loss = T.mean((X_reconstructed - X_batch) ** 2)
        
        params = layers.get_all_params(self.net)
        
        learning_rate = T.fscalar('learning rate')
        upd_log = updates.adadelta(log_loss, params, learning_rate=learning_rate)
        upd_mse = updates.adadelta(mse_loss, params, learning_rate=learning_rate)
        
        self.train_log = theano.function([X_batch, learning_rate], log_loss, updates=upd_log)
        self.train_mse = theano.function([X_batch, learning_rate], mse_loss, updates=upd_mse)
        
        self.encode = theano.function([X_batch], code)
        
        code_given = T.fmatrix('given code')
        X_decoded = layers.get_output(self.net, inputs={ dense1 : code_given })
        self.decode = theano.function([code_given], X_decoded)
        
        self.reconstruct = theano.function([X_batch], X_reconstructed)
    
    def fit(self, X, n_epoches, batch_size, learning_rate=1.0, loss='mse', watcher=None):
        n_batches = X.shape[0] / batch_size

        losses = np.zeros(shape=(n_epoches, n_batches))
        
        if loss == 'log':
            train = self.train_log
        elif loss == 'mse':
            train = self.train_mse
        else:
            raise Exception('Unknown loss!')
        
        learning_rate = np.float32(learning_rate)
        
        for epoch in xrange(n_epoches):
            indx = np.random.permutation(X.shape[0])
            
            for batch in xrange(n_batches):
                batch_indx = indx[(batch * batch_size):((batch + 1) * batch_size)]
                losses[epoch, batch] = train(X[batch_indx], learning_rate)
            
            if watcher is not None:
                watcher.draw(losses[:(epoch + 1)])

        return losses