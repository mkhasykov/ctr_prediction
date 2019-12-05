from csv import reader
from math import exp, sqrt
from random import random

from xxhash import xxh64


def data(path, D):
    ''' GENERATOR: 
            Apply hash-trick to the original csv row
            and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''
    
    with open(path, 'r', encoding='utf-8') as f:
        csvreader = reader(f)  # create a CSV reader
        header = next(csvreader)
        for row in csvreader:  # iterate over the available rows
            row = dict(zip(header, row))
            
            # ts and bid_id are used only while updating train data
            for feat in ['bid_id', 'ts']:
                if feat in row:
                    del row[feat]
            
            # process clicks
            y = 0.
            target='click'
            if target in row:
                if row[target] == '1':
                    y = 1.
                del row[target]
    
            # build x
            x = []
            for key in row:
                value = row[key]
                # one-hot encode everything with hash trick
                index = xxh64(key + '_' + value).intdigest() % D
                x.append(index)
    
            yield x, y


class ftrl_proximal(object):
    ''' Main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [random() for k in range(D)] #[0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in range(L):
                for j in range(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield xxh64(str(x[i]) + '_' + str(x[j])).intdigest() % D

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = round((sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2), 5)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            # if there were too many gradient steps along the feature
            # don't do step
            if (abs(z[i]) > 10e15)|(n[i] > 10e30):
                continue
            else:
                sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
                z[i] += g - sigma * w[i]
                n[i] += g * g
            
    def fit(self, path, epoch_num):
        ''' Fit model on a bunch of training data

            INPUT:
                path: path to training file
                epoch_num: number of training epochs

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''
        for e in range(epoch_num):
            for x, y in data(path, self.D):  # data is a generator
                p = self.predict(x)
                self.update(x, p, y)
    
    def test(self, path):
        ''' Get target values and corresponding prediction for a bunch of test data

            INPUT:
                path: path to test file

            OUTPUT:
                ys: list of target values
                preds: list of prediction values
        '''
        preds = []
        ys    = []
        for x, y in data(path, self.D):
            p = self.predict(x)
            preds += [p]
            ys    += [y]
        return ys, preds
    
    def output_weigts(self):
        ''' Build the complete weight vector (for following saving)

            OUTPUT:
                w: weight vector for logistic regression
        '''
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2
        
        # model
        n = self.n
        z = self.z
        w = {}
        for i in range(len(z)):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]
        
            # build the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = round((sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2), 5)
        
        return w
    
