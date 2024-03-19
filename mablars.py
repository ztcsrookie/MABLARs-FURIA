import copy

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import weka.core.jvm as jvm

from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.ANM.ANM import ANM
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import fisherz, chisq
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from weka.core.converters import Loader, load_any_file
from weka.classifiers import Classifier
from weka.filters import Filter
from weka.classifiers import Evaluation
from weka.core.dataset import create_instances_from_matrices



class CausalGraph:
    def __init__(self, causal_matrix=None, causal_dag=None, node_name_list=None, png_save_path='test_png'):
        '''
        :param causal_matrix: If A[i,j] !=0, the causal relationship: i -> j
        :param causal_dag: The causal graph object of the causal-learn library
        :param node_name_list: The node name list
        '''
        self.causal_matrix = causal_matrix
        self.causal_dag = causal_dag
        self.node_name_list = node_name_list

    def fit_ica(self, le_data, current_seeds=3):
        cd_model = lingam.ICALiNGAM(random_state=current_seeds, max_iter=10000)
        cd_model.fit(le_data)
        weighted_causal_matrix = copy.deepcopy(cd_model.adjacency_matrix_)
        self.causal_matrix = weighted_causal_matrix.T
        self.causal_dag = create_dag_from_matrix(self.causal_matrix, self.node_names_list)


    def fit_pc_anm(self, le_data):
        CG = pc(le_data, 0.05, fisherz, True, 1, 2)
        cur_graph_matrix = copy.deepcopy(CG.G.graph)
        D = cur_graph_matrix.shape[0]
        # Determine the direction of the undirected edge.
        for i in range(D):
            for j in range(i):
                if cur_graph_matrix[i, j] == 0:
                    continue
                elif cur_graph_matrix[i, j] == cur_graph_matrix[j, i]:
                    data_x = le_data[:, i].reshape(-1, 1)
                    data_y = le_data[:, j].reshape(-1, 1)
                    anm = ANM()
                    p_value_foward, p_value_backward = anm.cause_or_effect(data_x, data_y)
                    if p_value_foward > p_value_backward:  # i-->j
                        cur_graph_matrix[i, j] = -1
                        cur_graph_matrix[j, i] = 1
                    else:
                        cur_graph_matrix[i, j] = 1
                        cur_graph_matrix[j, i] = -1
        self.causal_matrix = cg_matrix_to_adjacent_matrix(cur_graph_matrix)
        self.causal_dag = create_dag_from_matrix(self.causal_matrix, self.node_name_list)


def create_dag_from_matrix(causal_matrix, node_names_list):
    nodes = []
    causal_matrix
    for name in node_names_list:
        node = GraphNode(name)
        nodes.append(node)

    dag = Dag(nodes)

    num_variables = causal_matrix.shape[0]

    for i in range(num_variables):
        for j in range(num_variables):
            if causal_matrix[i, j] != 0:
                dag.add_directed_edge(nodes[i], nodes[j])

    return dag


def cg_matrix_to_adjacent_matrix(A):
    """
    Convert causal graph matrix where A[i,j] == -1 and A[j,i] == 1, then i->j to
    a matrix B where B[i,j] == 1 then i->j, otherwise 0.
    :param A: The causal graph matrix
    :return: B
    """
    B = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == -1 and A[j, i] == 1:
                B[i, j] = 1
    return B


def find_mb(causal_matrix, x=-1) -> list:
    # 获取矩阵的维度
    num_var = causal_matrix.shape[0]
    # 1. 获取X的所有父节点
    parents_of_X = [i for i in range(num_var) if causal_matrix[i, x] != 0]

    # 2. 获取X的所有子节点
    children_of_X = [i for i in range(num_var) if causal_matrix[x, i] != 0]

    # 3. 获取X的所有子节点的其他父节点
    spouses_of_X = []
    for child in children_of_X:
        spouses_of_X.extend([i for i in range(num_var) if causal_matrix[i, child] != 0 and i != x])

    # 合并所有的组件并去除重复的以及目标变量，即最后一个变量
    markov_blanket = set(parents_of_X + children_of_X + spouses_of_X)
    markov_blanket.discard(num_var - 1)

    return list(markov_blanket)


def find_mbcd(causal_matrix, x=-1) -> list:
    num_var = causal_matrix.shape[0]
    mbcd = []
    for i in range(num_var):
        if causal_matrix[i, x] != 0:
            mbcd.append(i)
    if not mbcd:
        return []
    else:
        mbcd_set = set(mbcd)
        if num_var - 1 in mbcd_set:
            final_mbcd = list(mbcd_set.discard(num_var - 1))
        else:
            final_mbcd = list(mbcd_set)
        return final_mbcd


def show_causal_graph(save_path, causal_dag):
    """
    Show the causal graph and save to the save path.
    :param causal_dag: The direct causal graph, dag of the causal-learn package.
    :param save_path: The save path.
    :return:
    """
    graphviz_dag = GraphUtils.to_pgv(causal_dag)
    graphviz_dag.draw(save_path, prog='dot', format='png')
    img = mpimg.imread(save_path)
    plt.axis('off')
    plt.imshow(img)
    plt.show()


class SimpleMamdaniFuzzySystem:
    def __init__(self, rule_base=None, linguictic_rule_base=None, fuzzy_sets_parameters=None):
        '''
        :param rule_base: numpy matrix, R*D, R: #rules, D:#variables
        :param fuzzy_sets_parameters: numpy matrix, P*D, P: #fuzzy sets of each variable. D:#variables
        '''
        self.rule_base = rule_base
        self.linguistic_rule_base = linguictic_rule_base
        self.fuzzy_sets_parameters = fuzzy_sets_parameters

    def predict(self, x) -> int:
        '''

        :param x: an input sample
        :return: y: the predicted label
        '''
        n_rules = self.rule_base.shape[0]
        D = self.rule_base.shape[1] - 1
        md_rules = np.ones((n_rules, D))
        for r in range(n_rules):
            for d in range(D):
                # print(x[d])
                a = self.fuzzy_sets_parameters[0, d]
                b = self.fuzzy_sets_parameters[1, d]
                c = self.fuzzy_sets_parameters[2, d]
                if self.rule_base[r, d] == 0:
                    md_rules[r, d] = left_shoulder_mf(x[d], a, b)
                elif self.rule_base[r, d] == 1:
                    md_rules[r, d] = triangle_mf(x[d], a, b, c)
                else:
                    md_rules[r, d] = right_shoulder_mf(x[d], b, c)
        prod_md_rules = np.prod(md_rules, axis=1, keepdims=True)
        max_row_index = np.argmax(prod_md_rules)

        predict_y = self.rule_base[max_row_index, -1]
        return predict_y

    def wm_fit(self, x, y, n_clusters=3) -> None:
        '''

        :param x: the features, nparray N*D, N: #samples, D: #features
        :param y: the label. nparray N*1. N: #samples,
        :return:
        '''
        N, D = x.shape

        # u, cntr, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        #     x, n_clusters, 1.5, error=0.00001, maxiter=10000, init=None)

        kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=20, max_iter=1000, tol=1e-6, algorithm='elkan')
        kmeans.fit(x)
        cntr = kmeans.cluster_centers_

        self.fuzzy_sets_parameters = np.sort(cntr, axis=0)
        rule_base_dict = {}
        linguistic_rule_base = {}
        md_dict = {}

        for idx, sample in enumerate(x):
            current_product_md = 1
            candidate_rule = ()
            candidate_linguistic_rule = ()
            for i, feature_value in enumerate(sample):
                a = self.fuzzy_sets_parameters[0, i]
                b = self.fuzzy_sets_parameters[1, i]
                c = self.fuzzy_sets_parameters[2, i]
                md_low = left_shoulder_mf(feature_value, a, b)
                md_medium = triangle_mf(feature_value, a, b, c)
                md_high = right_shoulder_mf(feature_value, b, c)
                memberships = [
                    (md_low, 0, "Low"),
                    (md_medium, 1, "Mid"),
                    (md_high, 2, "High")
                ]
                max_md_i = max([md[0] for md in memberships])
                current_product_md *= max_md_i
                antecedent = max(memberships, key=lambda item: item[0])[1]
                linguistic_antecedent = max(memberships, key=lambda item: item[0])[2]
                candidate_rule += (antecedent,)
                candidate_linguistic_rule += (linguistic_antecedent,)

            if candidate_linguistic_rule not in linguistic_rule_base:
                linguistic_rule_base[candidate_linguistic_rule] = y[idx]
                rule_base_dict[candidate_rule] = y[idx]
                md_dict[candidate_linguistic_rule] = current_product_md

            elif candidate_linguistic_rule in linguistic_rule_base and current_product_md > md_dict[
                candidate_linguistic_rule]:
                linguistic_rule_base[candidate_linguistic_rule] = y[idx]
                rule_base_dict[candidate_rule] = y[idx]
                md_dict[candidate_linguistic_rule] = current_product_md

        numeric_rule_base = np.zeros((len(rule_base_dict), D + 1))
        r = 0
        for key, value in rule_base_dict.items():
            for i, idx in enumerate(key):
                numeric_rule_base[r, i] = idx
            numeric_rule_base[r, -1] = value
            r += 1
        self.rule_base = numeric_rule_base
        self.linguistic_rule_base = linguistic_rule_base


def triangle_mf(x, a, b, c):
    '''


    :param x: the input value
    :param a: the left value of the triangle membership function
    :param b: the middle value of the triangle membership function
    :param c: the right value of the triangle membership function
    :return: md_x: the membership degree of x to the triangle membership function
    '''
    if x<=a:
        md_x = 0
    elif a<x<=b:
        md_x = (x-a)/(b-a)
    elif b<x<=c:
        md_x = (c-x)/(c-b)
    else:
        md_x = 0
    return md_x

def left_shoulder_mf(x,a,b):
    '''
    下降那个
    :param x: the input value
    :param a: the left value of the left shoulder membership function
    :param b: the right value of the left shoulder membership function
    :return: md_x: the membership degree of x to the left shoulder membership function
    '''
    if x<=a:
        md_x = 1
    elif a<x<b:
        md_x = (b-x)/(b-a)
    else:
        md_x = 0
    return md_x

def right_shoulder_mf(x,a,b):
    '''
    上升那个
    :param x: the input value
    :param a: the left value of the right shoulder membership function
    :param b: the right value of the right shoulder membership function
    :return: md_x: the membership degree of x to the right shoulder membership function
    '''
    if x<=a:
        md_x = 0
    elif a<x<b:
        md_x = (x-a)/(b-a)
    else:
        md_x = 1
    return md_x

def MABLAR(x, original_y, node_name_list, mb_or_mbcd=0):
    '''

    :param x:the input data, a numpy matrix, n*d, n is number of samples, d is the number of inpout variables
    :param original_y: the original y, maybe a string, a n*1 numpy matrix
    :param node_name_list: the node name, a list
    :param mb_or_mbcd: default 0, if 1, then MBCD,
    :return:
    '''
    label_encoder = LabelEncoder()
    encoded_y = label_encoder.fit_transform(original_y)
    encoded_y = encoded_y.reshape(-1,1)
    encoded_data = np.concatenate((x,encoded_y),axis=1)
    causal_graph = CausalGraph(node_name_list=node_name_list)
    causal_graph.fit_pc_anm(encoded_data)
    mb = find_mb(causal_graph.causal_matrix)
    mbcd = find_mbcd(causal_graph.causal_matrix)
    if mb_or_mbcd == 0:
        input_x = x[:, mb]
    elif mb_or_mbcd ==1:
        input_x = x[:, mbcd]
    fuzzy_system = SimpleMamdaniFuzzySystem()
    fuzzy_system.wm_fit(input_x,encoded_y)
    return causal_graph, fuzzy_system



def load_csv_data(data_path):
    with open(data_path, 'r') as f:
        node_name_list = f.readline().strip().split(',')
        num_features = len(node_name_list) - 1
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)

    x = np.genfromtxt(data_path, delimiter=',', skip_header=1, usecols=range(num_features))
    original_y = np.genfromtxt(data_path, delimiter=',', skip_header=1, usecols=(num_features), dtype=str)
    return x, original_y, node_name_list

# def load_arff_data(arff_data_path):
#     arff_loader = Loader("weka.core.converters.ArffLoader")
#     arff_data = arff_loader.load_file(arff_data_path)
#     arff_data.class_is_last()
#
#     # check whether the class attribute is numeric or not
#     class_attribute = arff_data.class_attribute
#     is_numeric = class_attribute.is_numeric
#
#     # if the class attribute is numeric, discrete them
#     if is_numeric:
#         # reload the data set
#         arff_loader = Loader("weka.core.converters.ArffLoader")
#         arff_data_set_path = 'ARFF_Datasets/' + data_set_name + '.arff'
#         arff_data = arff_loader.load_file(arff_data_set_path)
#
#         # discrete the labels
#         discretize = Filter(classname="weka.filters.unsupervised.attribute.Discretize")
#         discretize.inputformat(arff_data)
#         arff_data = discretize.filter(arff_data)
#         arff_data.class_is_last()
#     return arff_data

def load_arff_data(arff_data_path):
    arff_data = load_any_file(arff_data_path)
    arff_data.class_is_last()
    return arff_data

def FURIA_train(train_data):
    furia = Classifier(classname="weka.classifiers.rules.FURIA")
    furia.build_classifier(train_data)
    return furia

def FURIA_test(model, test_data):
    evaluation = Evaluation(test_data)
    evaluation.test_model(model, test_data)
    return evaluation

def load_csv_data_for_furia(data_set_path):
    x, original_y, node_name_list = load_csv_data(data_set_path)
    dataset = create_instances_from_matrices(x, original_y, name="generated from matrices", cols_x=node_name_list[:-1],
                                             col_y=node_name_list[-1])
    dataset.class_is_last()
    return dataset

def create_mb_subset_for_furia(original_data_set, mb_index, node_name_list):

    mb_features = [node_name_list[i] for i in mb_index]
    mb_features.append(node_name_list[-1])

    subset = original_data_set.subset(col_names=mb_features)
    subset.class_is_last()
    return subset