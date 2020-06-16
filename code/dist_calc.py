from math import sqrt


def man_dist(point1, point2):
    return (abs(point1[0] - point2[0]) + abs(point1[1] - point2[1]))


def euclidean_dist(point1, point2):
    x_diff = point1[0] - point2[0]
    y_diff = point1[1] - point2[1]
    x_diff_by_2 = x_diff ** 2
    y_diff_by_2 = y_diff ** 2
    euclid_dist = sqrt(x_diff_by_2 + y_diff_by_2)
    return euclid_dist


def frechet_dist(point1, point2):
    x_diff = abs(point1[0] - point2[0])
    y_diff = abs(point1[1] - point2[1])
    return max(x_diff, y_diff)
