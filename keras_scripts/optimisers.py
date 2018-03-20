from keras.optimizers import Adam, SGD


def adam(lr=0.0001):
    return Adam(lr=lr)


def sgd(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True):
    return SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
