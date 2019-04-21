import numpy as np
import random


''' x is n by 1 numpy vector
    y is also n by 1 numpy vector
'''


def predict(w, b, x):
    predict_y = w * x + b
    return predict_y


def eval_loss(w, b, x, y):
    predict_y = predict(w, b, x)
    loss = ((predict_y - y)**2).sum() / (2 * len(y))
    return loss


def gradient(predict_y, batch_x, batch_y):
    batch_size = len(batch_x)
    diff = predict_y - batch_y
    dw = diff * batch_x
    db = diff
    dw = dw.sum() / batch_size
    db = db.sum() / batch_size
    return dw, db


def step_gradient(batch_x, batch_y, w, b, lr):
    predict_y = predict(w, b, batch_x)
    dw, db = gradient(predict_y, batch_x, batch_y)
    # update
    w -= lr * dw
    b -= lr * db
    return w, b


def train(x_list, y_list, batch_size, lr, max_iter):
    w = 0
    b = 0
    num_samples = len(x_list)
    for i in range(max_iter):
        batch_idxs = np.random.choice(num_samples, batch_size)
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [y_list[j] for j in batch_idxs]
        # convert list to ndarray
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        w, b = step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0}, b:{0}'.format(w, b))
        print('loss is {0}'.format(eval_loss(w, b, np.array(x_list), np.array(y_list))))


def gen_sample_data():
    w = random.randint(0, 10) + random.random()
    b = random.randint(0, 5) + random.random()
    num_samples = 100
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = random.randint(0, 100) * random.random()
        y = w * x + b + random.random() * random.random()
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list


def main():
    x_list, y_list = gen_sample_data()
    batch_size = 50
    lr = 0.001
    max_iter = 10000
    train(x_list, y_list, batch_size, lr, max_iter)


if __name__ == '__main__':
    main()