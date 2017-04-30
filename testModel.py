import cPickle as pickle
import theano
import numpy
import sys
import os

from trainMLP import MLP, LogisticRegression, HiddenLayer

def load_data_raw(dataset):
    with open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    return [train_set, valid_set, test_set]

def load_data(dataset):
    with open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
            
    def shared_dataset(data_xy, borrow=True):
            """ Function that loads the dataset into shared variables

            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch everytime
            is needed (the default behaviour if the data is not in a shared
            variable) would lead to a large decrease in performance.
            """
            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            # return shared_x, T.cast(shared_y, 'int32')
            return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


if __name__ == '__main__':
    datasetsRaw = load_data_raw(dataset='textureHistograms.pkl')
    nBatches = len(datasetsRaw[2][0])/20
    datasets = load_data(dataset='textureHistograms.pkl')
    x_test, y_test = datasets[2]
    
    modelFilename = sys.argv[1]
    with open(modelFilename, 'rb') as f:
        model = pickle.load(f)    
    
    errors = []
    predictions = []
    batch_size=20
    test_fn = model.test_function(x_test, y_test, batch_size)
    for i in xrange(nBatches):
        error, output_y = test_fn(i)
        errors.append(error)
        predictions.append(output_y)
        
    print numpy.mean(errors)
    with open('predictions/{}'.format(os.path.basename(modelFilename)), 'wb') as f:
        pickle.dump(predictions, f)
