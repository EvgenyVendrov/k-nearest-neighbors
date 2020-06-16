import dist_calc
from statistics import mean


def get_avg_error_by_k_and_p(dict_of_accuracy_by_k_and_p):
    list_of_acc_avg = []
    for k in range(1, 10, 2):
        for p in range(1, 4):
            list_of_acc_avg.append(cut_the_dict_and_calc_avg(dict_of_accuracy_by_k_and_p, "%dk_%dp" % (k, p)))

    # dict_of_1k_1p = {k: v for k, v in dict_of_accuracy_by_k_and_p.items() if k.startswith("1k_1p")}
    # dict_of_1k_2p = {k: v for k, v in dict_of_accuracy_by_k_and_p.items() if k.startswith("1k_2p")}
    # dict_of_1k_3p = {k: v for k, v in dict_of_accuracy_by_k_and_p.items() if k.startswith("1k_3p")}
    # dict_of_3k_1p = {k: v for k, v in dict_of_accuracy_by_k_and_p.items() if k.startswith("3k_1p")}
    # dict_of_3k_2p = {k: v for k, v in dict_of_accuracy_by_k_and_p.items() if k.startswith("3k_2p")}
    # dict_of_3k_3p = {k: v for k, v in dict_of_accuracy_by_k_and_p.items() if k.startswith("3k_3p")}
    # dict_of_5k_1p = {k: v for k, v in dict_of_accuracy_by_k_and_p.items() if k.startswith("5k_1p")}
    # dict_of_5k_2p = {k: v for k, v in dict_of_accuracy_by_k_and_p.items() if k.startswith("5k_2p")}
    # dict_of_5k_3p = {k: v for k, v in dict_of_accuracy_by_k_and_p.items() if k.startswith("5k_3p")}
    # dict_of_7k_1p = {k: v for k, v in dict_of_accuracy_by_k_and_p.items() if k.startswith("7k_1p")}
    # dict_of_7k_2p = {k: v for k, v in dict_of_accuracy_by_k_and_p.items() if k.startswith("7k_2p")}
    # dict_of_7k_3p = {k: v for k, v in dict_of_accuracy_by_k_and_p.items() if k.startswith("7k_3p")}
    # dict_of_9k_1p = {k: v for k, v in dict_of_accuracy_by_k_and_p.items() if k.startswith("9k_1p")}
    # dict_of_9k_2p = {k: v for k, v in dict_of_accuracy_by_k_and_p.items() if k.startswith("9k_2p")}
    # dict_of_9k_3p = {k: v for k, v in dict_of_accuracy_by_k_and_p.items() if k.startswith("9k_3p")}
    #
    # avg_of_1k_1p = mean(dict_of_1k_1p.values())
    # avg_of_1k_2p = mean(dict_of_1k_2p.values())
    # avg_of_1k_3p = mean(dict_of_1k_3p.values())
    # avg_of_3k_1p = mean(dict_of_3k_1p.values())
    # avg_of_3k_2p = mean(dict_of_3k_2p.values())
    # avg_of_3k_3p = mean(dict_of_3k_3p.values())
    # avg_of_5k_1p = mean(dict_of_5k_1p.values())
    # avg_of_5k_2p = mean(dict_of_5k_2p.values())
    # avg_of_5k_3p = mean(dict_of_5k_3p.values())
    # avg_of_7k_1p = mean(dict_of_7k_1p.values())
    # avg_of_7k_2p = mean(dict_of_7k_2p.values())
    # avg_of_7k_3p = mean(dict_of_7k_3p.values())
    # avg_of_9k_1p = mean(dict_of_9k_1p.values())
    # avg_of_9k_2p = mean(dict_of_9k_2p.values())
    # avg_of_9k_3p = mean(dict_of_9k_3p.values())
    #
    # return (
    #     avg_of_1k_1p, avg_of_1k_2p, avg_of_1k_3p, avg_of_3k_1p, avg_of_3k_2p, avg_of_3k_3p, avg_of_5k_1p, avg_of_5k_2p,
    #     avg_of_5k_3p, avg_of_7k_1p, avg_of_7k_2p, avg_of_7k_3p, avg_of_9k_1p, avg_of_9k_2p, avg_of_9k_3p)
    return list_of_acc_avg


def get_acc_on_train_set(train_set, test_set, k, p):
    list_of_labels = label_whole_set(train_set, test_set, k, p)
    acc = 0
    for actual_point, label_from_model in zip(test_set, list_of_labels):
        if label_from_model == actual_point[2]:
            acc += 1
    return acc / len(test_set)


def label_whole_set(train_set, test_set, k, p):
    list_of_labels = []
    for test_point in test_set:
        if p == 1:
            k_neighbors = get_k_nearest_neighbors(train_set, test_point, k, dist_calc.man_dist)
        elif p == 2:
            k_neighbors = get_k_nearest_neighbors(train_set, test_point, k, dist_calc.euclidean_dist)
        else:
            k_neighbors = get_k_nearest_neighbors(train_set, test_point, k, dist_calc.frechet_dist)
        list_of_labels.append(label_point(k_neighbors))

    return list_of_labels


def get_k_nearest_neighbors(train_set, datapoint_to_check, k, func_to_calc_dist):
    dict_of_dist_between_data_p_train_p = {}
    for train_data_point in train_set:
        dict_of_dist_between_data_p_train_p[train_data_point] = func_to_calc_dist(train_data_point, datapoint_to_check)
    sorted_dict_of_dist_between_data_p_train_p = {k: v for k, v in sorted(dict_of_dist_between_data_p_train_p.items(),
                                                                          key=lambda item: item[1])}
    return list(sorted_dict_of_dist_between_data_p_train_p)[:k]


def label_point(k_neighbors):
    # for x in k_neighbors:
    #     print(x[2])
    a_labeled = sum(x[2] == 1 for x in k_neighbors)
    b_labeled = sum(x[2] == -1 for x in k_neighbors)
    # print("a=>",a_labeled)
    # print("b=>",b_labeled)
    return 1 if a_labeled > b_labeled else -1


def cut_the_dict_and_calc_avg(dict_of_accuracy_by_k_and_p, value):
    dict_of_acc = {k: v for k, v in dict_of_accuracy_by_k_and_p.items() if k.startswith(value)}
    return mean(dict_of_acc.values())
