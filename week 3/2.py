import numpy as np
import random
import matplotlib.pyplot as plt

''' Logistic Regression'''

err_vec = []  # error
train_acc = []  # train accuracy


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def hypothesis(w1, w2, b, x):
    h = w1*x[:, 0] + w2*x[:, 1] + b
    h = sigmoid(h)
    return h.reshape((len(h), 1))


def eval_loss(w1, w2, b, x, y):
    h = hypothesis(w1, w2, b, x)
    loss = - y * np.log(h) - (1 - y) * np.log(1 - h)
    return loss.sum() / len(loss)


def gradient(h, batch_x, batch_y):
    batch_size = len(batch_x)
    diff = h - batch_y
    dw1 = diff * batch_x[:, 0]
    dw2 = diff * batch_x[:, 1]
    db = diff
    dw1 = dw1.sum() / batch_size
    dw2 = dw2.sum() / batch_size
    db = db.sum() / batch_size
    return dw1, dw2, db


def step_gradient(batch_x, batch_y, w1, w2, b, lr):
    h = hypothesis(w1, w2, b, batch_x)
    dw1, dw2, db = gradient(h, batch_x, batch_y)
    # update
    w1 -= lr * dw1
    w2 -= lr * dw2
    b -= lr * db
    return w1, w2, b


def train(x_train, y_train, batch_size, lr, max_iter):
    w1 = random.random()
    w2 = random.random()
    b = random.random()

    num_samples = len(y_train)
    for i in range(max_iter):
        batch_idxs = np.random.choice(num_samples, batch_size)
        batch_x = x_train[batch_idxs]
        batch_y = y_train[batch_idxs]
        w1, w2, b = step_gradient(batch_x, batch_y, w1, w2, b, lr)

        err = eval_loss(w1, w2, b, x_train, y_train)
        err_vec.append(err)
        predict_y = hypothesis(w1, w2, b, x_train)
        predict_y = (predict_y > 0.5)
        predict_y = predict_y.astype(np.float32)
        acc = (predict_y == y_train).astype(np.float32).sum() / len(y_train)
        train_acc.append(acc)
        # print something
        if i % 100 == 0:
            print('i:{0}, w1:{1}, w2:{2}, b:{3}'.format(i, w1, w2, b))
            print('loss is {0}, training accuracy is: {1}'.format(err, acc))
    return w1, w2, b


def gen_sample_data():
    # n by 2
    num_samples = 2000
    sample_var = 100
    data1 = np.random.multivariate_normal([50, 30], [[sample_var, 0], [0, sample_var]], num_samples//2)
    data2 = np.random.multivariate_normal([90, 10], [[sample_var, 0], [0, sample_var]], num_samples//2)
    x_train = np.concatenate((data1, data2), axis=0)
    # label
    y_train = np.zeros((num_samples, 1))
    y_train[num_samples//2:, :] = 1
    # generate testing data
    num_testing = 200
    data3 = np.random.multivariate_normal([50, 30], [[sample_var, 0], [0, sample_var]], num_testing//2)
    data4 = np.random.multivariate_normal([90, 10], [[sample_var, 0], [0, sample_var]], num_testing//2)
    x_test = np.concatenate((data3, data4), axis=0)
    y_test = np.zeros((num_testing, 1))
    y_test[num_testing//2:, :] = 1
    return x_train, y_train, x_test, y_test


def normalized(v):
    v_std = np.std(v)
    v_mean = np.mean(v)
    v = (v - v_mean) / v_std
    return v


def main():
    # get data
    x_train, y_train, x_test, y_test = gen_sample_data()
    # normalized data
    x_train[:, 0] = normalized(x_train[:, 0])
    x_train[:, 1] = normalized(x_train[:, 1])
    x_test[:, 0] = normalized(x_test[:, 0])
    x_test[:, 1] = normalized(x_test[:, 1])
    # setting hyper parameters
    batch_size = len(y_train) // 2
    lr = 0.01
    max_iter = 1000
    # train
    w1, w2, b = train(x_train, y_train, batch_size, lr, max_iter)
    print('Model w1: {0}, w2: {1}, w3:{2}'.format(w1, w2, b))
    # compute testing accuracy
    predict_y = hypothesis(w1, w2, b, x_test)
    predict_y = (predict_y > 0.5)
    predict_y = predict_y.astype(np.float32)
    test_acc = (predict_y == y_test).astype(np.float32).sum() / len(y_test)
    print('testing data accuracy: {0}'.format(test_acc))
    # plot error
    plt.figure(1)
    plt.plot(range(1, len(err_vec)+1), err_vec)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Loss vs. iteration')
    plt.figure(2)
    plt.plot(range(1, len(train_acc)+1), train_acc)
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.title('training accuracy')
    # plot training data and decision boundary
    plt.figure(3)
    plt.scatter(x_train[:len(y_train)//2, 0], x_train[:len(y_train)//2, 1], c='c', marker='o', label='Class 1')
    plt.scatter(x_train[len(y_train)//2:, 0], x_train[len(y_train)//2:, 1], c='r', marker='D', label='Class 2')
    x1 = np.arange(int(x_train.min()-1), int(x_train.max()+1))
    x2 = 1/w2 * (- b - w1*x1)
    plt.plot(x1, x2, 'g--', linewidth=2, label='Decision Boundary')
    plt.title('Normalized Training Data and Decision Boundary')
    plt.legend()
    # plot testing data decision boundary
    plt.figure(4)
    plt.scatter(x_test[:len(y_test)//2, 0], x_test[:len(y_test)//2, 1], c='c', marker='o', label='Class 1')
    plt.scatter(x_test[len(y_test)//2:, 0], x_test[len(y_test)//2:, 1], c='r', marker='D', label='Class 2')
    x1 = np.arange(int(x_test.min()-1), int(x_test.max()+1))
    x2 = 1/w2 * (- b - w1*x1)
    plt.plot(x1, x2, 'g--', linewidth=2, label='Decision Boundary')
    plt.title('Normalized Testing Data and Decision Boundary')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()


