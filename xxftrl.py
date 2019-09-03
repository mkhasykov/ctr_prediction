from csv import reader
from math import exp, log, sqrt
from random import random
from xxhash import xxh64

# ftrl parameters
alpha = .005  # learning rate
beta = .1     # smoothing parameter for adaptive learning rate
L1 = 0.       # L1 regularization, larger value means more regularized
L2 = 0.       # L2 regularization, larger value means more regularized
# feature/hash trick parameters
D = 2 ** 18             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions
# training process parameters
epoch = 1       # learn training data for N passes


def find_nth(string, substring, n):
    ''' FUNCTION: Find index of nth occurence of substring in string

        INPUT:
            string: string where to seek
            substring: string to seek
            n: needed number of occurence of substring to seek
    
        OUTPUT:
            logarithmic loss of p given y
    '''
    start = string.find(substring)
    while start >= 0 and n > 1:
        start = string.find(substring, start+len(substring))
        n -= 1
    return start

# TODO: try focal loss
# TODO: there are some papers on ctr estimation under consideration of win/loss bias (weinan zhang)
def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def data(path, D):
    ''' GENERATOR: Apply hash-trick to the original csv row
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
        # TODO: del t
        for t, row in enumerate(csvreader):  # iterate over the available rows
            row = dict(zip(header, row))
            
            try:
                ID = row['bid_id']
                del row['bid_id']
            except:
                pass
            
            # del ts (ts is used only while updating train data by external script)
            if 'ts' in row:
                del row['ts']
            
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
    
            yield ID, x, y


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
            for _, x, y in data(path, self.D):  # data is a generator
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
        IDs   = []
        preds = []
        ys    = []
        for ID, x, y in data(path, self.D):
            p = self.predict(x)
            IDs   += [ID]
            preds += [p]
            ys    += [y]
        return IDs, ys, preds
    
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


if __name__ == "__main__":
    import argparse
    import time
    import json
    start = time.time()
    
    '''Usage: python xxftrl.py -i input.csv -o output.json -n 1 -d 18 -alpha .005 -beta .1 -l1 0. -l2 0.'''
    parser = argparse.ArgumentParser(description="Expected ctr for a bid request.")
    parser.add_argument("-i", "--input", required=True, dest="input",
                        help="input file (csv) path")
    parser.add_argument("-o", "--output", required=True, dest="output",
                        help="output file (json) path")
    parser.add_argument("-n", "--num_epoch", type=int, required=False, default=1, dest="num_epoch",
                        help="number of train epochs. Default value is 1")
    parser.add_argument("-d", "--dim_hash", type=int, required=False, default=18, dest="d",
                        help="parameter determines number of OHE features as 2**d. Default value is 18." + 
                        "For sake of results adequacy, values less than 15 are prohibited.")
    parser.add_argument("-alpha", "--alpha", type=float, required=False, default=0.005, dest="alpha", 
                        help="ftrl alpha parameter. Default value is .005")
    parser.add_argument("-beta", "--beta", type=float, required=False, default=0.1, dest="beta", 
                        help="ftrl beta parameter. Default value is .1")
    parser.add_argument("-l1", "--l1", type=float, required=False, default=0.0, dest="l1", 
                        help="L1 regularization parameter. Default value is 0.")
    parser.add_argument("-l2", "--l2", type=float, required=False, default=0.0, dest="l2", 
                        help="L2 regularization parameter. Default value is 0.")
    args = parser.parse_args()
    
    if args.num_epoch <= 0:
        raise Exception("Allowed only positive number of train epochs")
    if args.d < 15:
        raise Exception("Allowed hash feature space dimensionality not less than 2**15")
    if args.alpha < 0:
        raise Exception("Allowed only non-negative ftrl alpha parameter")
    if args.beta < 0:
        raise Exception("Allowed only non-negative ftrl beta parameter")
    if args.l1 < 0:
        raise Exception("Allowed only non-negative l1 regularization parameter")
    if args.l2 < 0:
        raise Exception("Allowed only non-negative l2 regularization parameter")
    
    D = 2**args.d # number of hash feature / weights
    
    # fit ftrl model
    learner = ftrl_proximal(args.alpha, args.beta, args.l1, args.l2, D, interaction)
    learner.fit(args.input, args.num_epoch)
    # form weights for online usage
    weights = learner.output_weigts()
    # save weights
    with open(args.output, 'w') as fp:
        json.dump(weights, fp)
    
    finish = time.time()
    print("TIME: {0:.2f}s".format(finish - start))
    
