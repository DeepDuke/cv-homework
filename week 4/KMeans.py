"""
Implement of K-Means Algorithm
I generate some data which consist of three clusters of normal distribution data.
I try to divide data into k clusters
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_clusters():
    """
    Generate three clusters
    :return: n by 2 , 2D matrix data points
    """
    np.random.seed(1)
    num_of_samples = 100
    mu1 = [0, 0]
    mu2 = [30, 30]
    mu3 = [60, 0]
    sigma = [[100, 0], [0, 100]]
    data1 = np.random.multivariate_normal(mu1, sigma, num_of_samples)
    data2 = np.random.multivariate_normal(mu2, sigma, num_of_samples)
    data3 = np.random.multivariate_normal(mu3, sigma, num_of_samples)
    # combine data together
    data = np.concatenate((data1, data2, data3), axis=0)
    # plot data
    plt.figure(1)
    plt.scatter(data[:, 0], data[:, 1], c='c', marker='o')
    plt.title('sample data')
    return data


def main(k):
    """
    Divide sample into k clusters using KMeans algorithm
    """
    sample = generate_clusters()
    num_sample = len(sample)
    class_label = [-1 for i in range(num_sample)]
    centers_id = np.random.choice(num_sample, k)
    centers = sample[centers_id]
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, k)]  # generate k sort of colors
    while True:  # centers do not change
        new_centers = np.zeros_like(centers)  # clear new_centers
        counts = [0*i for i in range(k)]
        print('new_centers shape', new_centers.shape)
        for i, point in enumerate(sample):  # get each data point's class label
            dis_vec = []
            for center in centers:  # compute distance between point and each center
                distance = np.linalg.norm(point - center)
                dis_vec.append(distance)
            # print('i', i, ' dis_vec:', dis_vec)
            label = dis_vec.index(min(dis_vec))  # 0, 1, 2
            # print(label)
            class_label[i] = label  # update label
            counts[label] += 1
            # print(class_label)
            new_centers[label, 0] += point[0]
            new_centers[label, 1] += point[1]
        # compute new centers
        for label in range(k):
            new_centers[label, 0] /= counts[label]
            new_centers[label, 1] /= counts[label]
        if (new_centers == centers).all():
            break
        centers = new_centers
    # draw centers and labeled data points
    plt.figure(2)
    # plot classified sample
    # print(class_label)
    for i, label in enumerate(class_label):
        point = sample[i]
        plt.scatter(point[0], point[1], c=colors[label], marker='o')
    # plot centers
    plt.scatter(centers[:, 0], centers[:, 1], s=200, c='g', marker='*', label='centroids')
    plt.title('Classification Results'+' When k={}'.format(k))
    return centers


if __name__ == '__main__':
    for K in range(2, 7):
        centroids = main(K)
        print(centroids)
        plt.legend()
        plt.show()
