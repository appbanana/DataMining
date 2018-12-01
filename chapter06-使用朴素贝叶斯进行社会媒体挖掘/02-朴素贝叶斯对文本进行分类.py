import numpy as np

"""
    参考文章：https://zhuanlan.zhihu.com/p/50287183
"""


def load_data_set():
    """
    :return: 将评论分离个单个的单词
    """
    cm1 = "my dog has flea problems help please"
    cm2 = "maybe not take him to dog park stupid"
    cm3 = "my dalmation is so cute I love him"
    cm4 = "stop posting stupid worthless garbage"
    cm5 = "mr licks ate my steak how to stop him"
    cm6 = "quit buying worthless dog food stupid"
    comments = [cm1, cm2, cm3, cm4, cm5, cm6]
    posting_list = []
    for comment in comments:
        posting_list.append(comment.split(' '))
    class_vec = [0, 1, 0, 1, 0, 1]  # 分类标签（0表示好的情况， 1表示坏的情况）
    return posting_list, class_vec


def create_vocab_list(data):
    """
    将输入数据集去重 排序 输出
    :param data:
    :return: 所有无重复的单词,并按照升序输出
    """
    world_set = set()
    for item_list in data:
        # 两个集合求并集
        world_set = world_set | set(item_list)
    word_order_list = list(sorted(world_set))
    return word_order_list


def set_of_words_2_vec(word_set, word_arr):
    """
    把所有不重复的单词排成一行 每个单词就是一列，相当于特征 如果word_arr输入的数据有某个单词就在这个单词对应位置为1 否则置位0
    :param word_set: 所有不重复单词的集合列表
    :param word_arr: 输入的数据集 在这里是每一行单词
    :return: 返回多维数组
    """
    vec_list = [0] * len(word_set)
    for word in word_arr:
        if word in word_set:
            # 若输入集中的单词出现在训练集中，则标记为1， 否则为0
            vec_list[word_set.index(word)] = 1
    # print(vec_list)
    return vec_list


def train_nb0(train_matrix, train_category):
    # 总共6条 没一行为一条数据
    num_of_traindocs = len(train_matrix)
    # 每一行有多少单词量 32个
    num_of_words = len(train_matrix[0])
    # 计算例子中侮辱性词条概率 3 / 6
    p_abusive = sum(train_category) / float(num_of_traindocs)
    """
        本程序用了贝叶斯估计来优化极大似然估计。因为极大似然估计一旦出现概率为0的时候就会使其他的所有乘法的结果都为0.所以为
        避免这种情况，在分子，分母上加上辅助项可以避免此类情况。 
        在此项目中分子加1(即下面使用的np.ones)，分母加类别总数2（ps：p0_denom,p1_denom初始化为2 此项目中只有2中类别，非侮辱性和侮辱性）
    """
    p0_num = np.ones(num_of_words)
    p1_num = np.ones(num_of_words)
    p0_denom = 2.0
    # 统计类别为侮辱性 所有输入集中单词总数
    p1_denom = 2.0
    for i in range(num_of_traindocs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            # 统计是侮辱性类别 单词总数
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    """
        p0_num 统计非侮辱性类别下 单词出现评率  是个32列一维数组 
            eg:[1, 2, 0, 0 ,..., 1]  0代表单词在非侮辱性类别出现0次 1代表单词在非侮辱性类别出现1次 2代表单词在非侮辱性类别出现2次 
        p0_denom 是非侮辱性类别下 出现单词的总量
        p0_num / p0_denom 是非侮辱性类别下出现的所有单词 每个单词出现的概率
        
        p1_vect，p1_denom， p1_num / p1_denom 与上面同义，只是类别为侮辱性类别
        因为p0_num / p0_denom相除后值很小，为了避免很多小的量相乘之后数值越乘越小导致下溢，所有取对数，乘法变加法 eg:log(A * B) = logA + logB
    """
    p0_vect = np.log(p0_num / p0_denom)
    p1_vect = np.log(p1_num / p1_denom)
    # p0_vect 是非侮辱性类别下每个单词出现的概率 即表示条件概率P(wi | c = 0)
    # p1_vect 是侮辱性类别下每个单词出现的概率 即表示条件概率P(wi | c = 1)
    # p_abusive 侮辱性类别出现的概率
    return p0_vect, p1_vect, p_abusive


def classify_nb(vec2_classify, p0_vec, p1_vec, p_class_1):
    """
    :param vec2_classify: 待测类型是数据集
    :param p0_vec: 是非侮辱性类别下每个单词出现的概率
    :param p1_vec:  是侮辱性类别下每个单词出现的概率
    :param p_class: p_class_1表示类标签为1时的概率P(c=1) 即侮辱性类别出现的概率
    :return:
    """

    """
    p(c|w) = p(w|c) * p(c) / p(w)
    p(w|c) = p(w1|c=0) * p(w2|c=0) * ... * p(w=i|c=0)
     对上面两个两边取对数
    log{p(w|c)} = log{p(w1|c=0)} + log{p(w2|c=0)} +....+log {p(wi|c=0)}

     log{p(c|w)} = log{p(w|c) * p(c)} / log{p(w)}
                 = log{p(w|c)} + log{p(c)} - log{p(w)}
                 = sum() + log{p(c)} - log{p(w)}  # 这就是下面式子的由来
    因为分母p(w) 是一样的，所以就没有参加计算  我还是有点不明白为什么分母就一样了呢？？？？
    """
    p0 = sum(vec2_classify * p0_vec) + np.log(1 - p_class_1)
    p1 = sum(vec2_classify * p1_vec) + np.log(p_class_1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    data_set, label = load_data_set()
    result = ["good comment", "bad comment"]
    vocab_List = create_vocab_list(data_set)
    # print(vocab_List)

    train_mat = []
    for world_list in data_set:
        train_mat.append(set_of_words_2_vec(vocab_List, world_list))
    # print(world_list)
    # print(train_mat)
    # print('\n')
    p0V, p1V, pAb = train_nb0(train_mat, label)

    # 测试部分
    # 非侮辱性信息
    # test_entry = "I love my dog very much"  # 输入测试集
    # 侮辱性信息
    test_entry = "stupid garbage dog"

    test_split = test_entry.split(' ')
    # 将输入转化为程序可以识别的格式
    this_doc = np.array(set_of_words_2_vec(vocab_List, test_split))
    result = classify_nb(this_doc, p0V, p1V, pAb)
    print(result)
