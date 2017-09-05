# encoding: utf-8

from math import log
import operator


def create_data_set():
    """
    创建数据集（训练集）
    :return:
    """
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = [u'能否浮出水面', u'是否有脚蹼']
    return data_set, labels


def calc_shannon_ent(data_set):
    """
    计算数据集的信息熵（熵越大，数据集的无序性越大）
    :param data_set:
    :return:
    """
    # 二维数组的行数
    num_entries = len(data_set)

    # 为所有可能的分类创建字典，格式为：{label: 出现该分类的次数}
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts:
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    # 香农熵计算公式
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def split_data_set(data_set, axis, value):
    """
    按照给定特征划分数据集
    (若指定的特征等于指定的value，则将该行提取出来，但是删除了该行这个特征)
    :param data_set: 待划分的数据集
    :param axis: 划分数据集的特征 的下标
    :param value: 特征的返回值
    :return:
    """
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    """
    选择最好的数据集划分方式（找到最好的划分特征）
    :param data_set:
    :return:
    """
    # 特征总数
    num_features = len(data_set[0]) - 1
    # 原始熵
    base_entropy = calc_shannon_ent(data_set)
    # 最大信息增益
    best_info_gain = 0.0
    # 用于划分数据集的最好特征
    best_feature = -1
    for i in range(num_features):
        feat_value_list = [example[i] for example in data_set]
        # 对于第i个特征，所有可能的取值
        unique_vals = set(feat_value_list)
        new_entropy = 0.0
        # 计算第i个特征能够带来的信息增益
        for value in unique_vals:
            # 第i个特征取值为value的所有行
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    """
    投票表决，返回票数最多的类别
    :param class_list:
    :return:
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count:
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    创建决策树
    :param data_set:
    :param labels:
    :return:
    """
    # 所有的分类
    class_list = [example[-1] for example in data_set]

    # 如果数据集中只有一种类别
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 如果数据集只有一列，也就是所有的标签都已经使用完毕；采用多数表决返回最多的类别
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)

    # 寻找最好的划分特征A
    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    # 从标签列表里删除该特征A
    del(labels[best_feat])

    # 找到该特征的所有可能值
    feat_values = [example[best_feat] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        # 对于去掉了特征A的数据集子集，递归构造树的子节点
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


if __name__ == '__main__':
    training_set, label_list = create_data_set()
    # print calc_shannon_ent(training_set)
    # print split_data_set(training_set, 0, 1)
    # print choose_best_feature_to_split(training_set)
    my_tree = create_tree(training_set, label_list)
    print my_tree

