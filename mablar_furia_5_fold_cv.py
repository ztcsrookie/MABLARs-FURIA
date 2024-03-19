import weka.core.jvm as jvm
import sys

from mablars import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from weka.core.classes import Random
from weka.filters import Filter


def split_train_test(arff_data, num_folds, fold):
    # creat stratified fold filter
    stratified_fold = Filter(
        classname="weka.filters.supervised.instance.StratifiedRemoveFolds",
        options=["-N", str(num_folds), "-F", str(fold + 1), "-S", "0", "-V"]
    )

    # create the training data set
    stratified_fold.inputformat(arff_data)
    train_data = stratified_fold.filter(arff_data)

    # create the testing data set
    stratified_fold.options = [
        "-N", str(num_folds), "-F", str(fold + 1), "-S", "0"
    ]
    stratified_fold.inputformat(arff_data)
    test_data = stratified_fold.filter(arff_data)
    return train_data, test_data


if __name__ == '__main__':
    jvm.start(packages=True)

    # The data set path of the csv data set
    data_set_path = 'CSV_Datasets/Pima_diabetes.csv'

    # Indicate which framework should be used
    mb_or_mbcd = 2  # 0--furia, 1--mb-furia, 2--mbcd-furia

    # Load the csv data set
    x, original_y, node_name_list = load_csv_data(data_set_path)

    # Convert the string label to nominal label
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(original_y)
    y = y.reshape(-1, 1)

    # Normalization
    scaler = MinMaxScaler()
    normalised_x = scaler.fit_transform(x)

    # Find the causal graph using the PC algorithm and the ANM algorithm
    data = np.concatenate((x, y), axis=1)
    causal_graph = CausalGraph(node_name_list=node_name_list)
    causal_graph.fit_pc_anm(data)

    # Show the obtained causal graph and save to the 'test.png'
    show_causal_graph('test.png', causal_graph.causal_dag)

    # Find the feature index of Markov blanket (mb_index) and direct causes (mbcd_index)
    mb_index = find_mb(causal_graph.causal_matrix)
    mbcd_index = find_mbcd(causal_graph.causal_matrix)

    # Determine which model should be used and create the corresponding subset
    if mb_or_mbcd == 0:
        normalised_input_x = normalised_x
        train_features_list = node_name_list
    elif mb_or_mbcd == 1:
        normalised_input_x = normalised_x[:, mb_index]
        train_features_list = [node_name_list[i] for i in mb_index]
        train_features_list.append(node_name_list[-1]) # mb_feature: the list of the name of each feature, including the name of the class feature
    else:
        if not mbcd_index:
            print('No direct cause')
            sys.exit(0)
        else:
            normalised_input_x = normalised_x[:, mbcd_index]
            train_features_list = [node_name_list[i] for i in mbcd_index]
            train_features_list.append(node_name_list[-1])

    # Create the arff data (the data format for the python-weke-wrapper3 package)
    arff_data = create_instances_from_matrices(normalised_input_x, original_y, name="generated from matrices",
                                                    cols_x=train_features_list[:-1], col_y=train_features_list[-1])
    string_to_nominal = Filter(classname="weka.filters.unsupervised.attribute.StringToNominal",
                               options=["-R", "last"])
    string_to_nominal.inputformat(arff_data)
    arff_data_nominal = string_to_nominal.filter(arff_data)
    # Indicate the class featrue
    arff_data_nominal.class_is_last()
    
    # 5-fold cross-validation
    num_folds = 5
    all_accuracy = []

    for fold in range(num_folds):
        train_data, test_data = split_train_test(arff_data_nominal, num_folds, fold)
        FURIA_model = FURIA_train(train_data)
        FURIA_result = FURIA_test(FURIA_model, test_data)
        all_accuracy.append(FURIA_result.percent_correct)

    # Show the result
    print(sum(all_accuracy)/len(all_accuracy))

    jvm.stop()