# encoding: utf-8

from numpy import *
import re
import operator


# 数据集，标签
def create_data_set():
    # 矩阵
    train_data = array([
        [0, 0.1],
        [0, 0],
        [1.0, 1.0],
        [1.0, 1.1]])
    labels = ['B', 'B', 'A', 'A']
    return train_data, labels


# KNN 分类的核心函数
def classify(in_X, data_set, labels, k):
    # 获取二维数组的行数
    data_set_size = data_set.shape[0]

    # 计算距离
    diff_mat = tile(in_X, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distance = sq_diff_mat.sum(axis=1)
    distances = sq_distance ** 0.5

    # 按距离排序
    sorted_dist_index = distances.argsort()

    # 统计前k个点所属点类别
    class_count = {}
    for i in range(k):
        label = labels[sorted_dist_index[i]]
        class_count[label] = class_count.get(label, 0) + 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 返回前k个点中频率最高的类别
    return sorted_class_count[0][0]


# 从文件中读取数据，转化为矩阵
def file2matrix(filename):
    fr = open(filename)
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)
    return_mat = zeros((number_of_lines, 4))
    class_label_vector = []
    index = 0
    for line in array_of_lines:
        line = line.strip()
        list_from_line = re.split('\s+', line)
        return_mat[index, :] = list_from_line[0:4]
        class_label_vector.append(str(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


# 特征值归一化
def auto_norm(data_set):
    """
    特征值归一化
    :param data_set:
    :return:
    """
    # 参数0使得函数选取列的最小值，而非行的最小值
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    # 行数
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


# 测试方法
def dating_class_test():
    # 使用10%的数据作为测试数据
    ho_ratio = 0.1
    dating_data_mat, dating_labels = file2matrix('./KNN/datingTestSet.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify(norm_mat[i, :], norm_mat[num_test_vecs: m, :],
                                     dating_labels[num_test_vecs:m], 3)
        print u'分类器结果：%d, 真实结果是：%d' % (classifier_result, dating_labels[i])
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print u'错误率为: %f' % (error_count / float(num_test_vecs))


if __name__ == '__main__':
    datingDataMat, datingLabels = file2matrix('./KNN/datingTestSet.txt')
    normMat, _, _ = auto_norm(datingDataMat)
    print classify(datingDataMat[12, :], normMat[0:11, :], datingLabels[0:11], 3)
