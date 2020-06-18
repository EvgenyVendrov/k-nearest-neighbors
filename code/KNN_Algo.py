import data_prep
import calculations


def knn(num_of_rounds=500, path_for_data_set=r"dataset\HC_Body_Temperature.txt"):
    (data_set) = data_prep.get_data_set_from_path(path_for_data_set)
    dict_of_test_accuracy_by_k_and_p = {}
    dict_of_train_accuracy_by_k_and_p = {}
    for i in range(0, num_of_rounds):
        (shuffled_train_set, shuffled_test_set) = data_prep.shuffle_dataset(data_set)
        for k in range(1, 10, 2):
            for p in range(1, 4):
                dict_of_test_accuracy_by_k_and_p["%dk_%dp_%d" % (k, p, i)] = calculations.get_acc_on_train_set(
                    shuffled_train_set,
                    shuffled_test_set,
                    k, p)
                dict_of_train_accuracy_by_k_and_p["%dk_%dp_%d" % (k, p, i)] = calculations.get_acc_on_train_set(
                    shuffled_train_set,
                    shuffled_train_set,
                    k, p)
    list_of_test_acc_avg = calculations.get_avg_error_by_k_and_p(dict_of_test_accuracy_by_k_and_p)
    list_of_train_acc_avg = calculations.get_avg_error_by_k_and_p(dict_of_train_accuracy_by_k_and_p)
    i = 0
    for k in range(1, 10, 2):
        for p in range(1, 4):
            print("TRAIN:")
            print_avg(k, p, list_of_train_acc_avg[i])
            print("TEST:")
            print_avg(k, p, list_of_test_acc_avg[i])
            i += 1


def print_avg(k, p, val):
    if p == 1:
        s = "avg of %d neighbors using manhattan dist is " % k
        print(s, val)
    elif p == 2:
        s = "avg of %d neighbors using euclidean dist is " % k
        print(s, val)
    else:
        s = "avg of %d neighbors using frechet dist is " % k
        print(s, val)


knn()
