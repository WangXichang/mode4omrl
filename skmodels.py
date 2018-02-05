# -*- utf8 -*-

class SklearnModel:

    classify_number = 2

    def __init__(self):
        self.sample_test_ratio = 0.85
        self.data_features = None
        self.data_labels = None
        self.test_features = None
        self.test_labels = None
        self.model_dict = {
            'bayes': SklearnModel.naive_bayes_classifier,
            'svm': SklearnModel.svm_classifier,
            'knn': SklearnModel.knn_classifier,
            'logistic_regression': SklearnModel.logistic_regression_classifier,
            'random_forest': SklearnModel.random_forest_classifier,
            'decision_tree': SklearnModel.decision_tree_classifier,
            'gradient_boosting': SklearnModel.gradient_boosting_classifier,
            'svm_cross': SklearnModel.svm_cross_validation,
            'kmeans': SklearnModel.kmeans_classifier,
            'mlp': SklearnModel.mlp_classifier
           }
        self.model = None
        self.test_result_labels = None
        self.model_test_result = dict({'suc_ratio':0, 'err_num':0})

    def set_data(self, data_feat, data_label):
        if data_label is not None:
            data_len = len(data_feat)
            train_len = int(data_len * self.sample_test_ratio)
            test_len = data_len - train_len
            self.data_features = data_feat[0:train_len]
            self.data_labels = data_label[0:train_len]
            self.test_features = data_feat[train_len:train_len + test_len]
            self.test_labels = data_label[train_len:train_len+test_len]
        else:
            self.data_features = data_feat

    def make_model(self, model_name='kmeans'):
        if model_name not in self.model_dict:
            # print('error model name:', model_name)
            return False
        if self.data_features is None:
            # print('data is not ready:', model_name)
            return False
        self.model = self.model_dict[model_name](self.data_features, self.data_labels)
        if self.test_labels is not None:
            self.test_result_labels = self.model.predict(self.test_features)
            sucnum = sum([1 if x == y else 0 for x,y in zip(self.test_labels, self.test_result_labels)])
            self.model_test_result['suc_ratio'] = sucnum / len(self.test_labels)
            self.model_test_result['err_num'] = len(self.test_labels) - sucnum
        return True

    def test_model(self, train_x, train_y):
        model_train_result = dict({'suc_ratio': 0, 'err_num': 0})
        test_result_labels = self.model.predict(train_x)
        test_result = [1 if x == y else 0 for x,y in zip(train_y, test_result_labels)]
        sucnum = sum(test_result)
        model_train_result['suc_ratio'] = sucnum / len(train_x)
        model_train_result['err_num'] = len(train_x) - sucnum
        model_train_result['err_feat'] = [{'feat': train_x[i], 'label': train_y[i], 'test_label': test_result_labels[i]}
                                          for i, x in enumerate(test_result) if x == 0]
        pp.pprint(model_train_result)

    def save_model(self, pathfile='model_name_xxx.m'):
        jb.dump(self.model, pathfile)

    @staticmethod
    # Multinomial Naive Bayes Classifier
    def kmeans_classifier(train_x, train_y):
        from sklearn.cluster import KMeans
        model = KMeans(SklearnModel.classify_number)
        if train_y is None:
            model.fit(train_x)
        else:
            model.fit(train_x, train_y)
        return model

    @staticmethod
    # Multinomial Naive Bayes Classifier
    def naive_bayes_classifier(train_x, train_y):
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB(alpha=0.01)
        model.fit(train_x, train_y)
        return model

    @staticmethod
    # KNN Classifier
    def knn_classifier(train_x, train_y):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
        model.fit(train_x, train_y)
        return model


    @staticmethod
    # Logistic Regression Classifier
    def logistic_regression_classifier(train_x, train_y):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(penalty='l2')
        model.fit(train_x, train_y)
        return model


    @staticmethod
    # Random Forest Classifier
    def random_forest_classifier(train_x, train_y):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=8)
        model.fit(train_x, train_y)
        return model


    @staticmethod
    # Decision Tree Classifier
    def decision_tree_classifier(train_x, train_y):
        from sklearn import tree
        model = tree.DecisionTreeClassifier()
        model.fit(train_x, train_y)
        return model


    @staticmethod
    # GBDT(Gradient Boosting Decision Tree) Classifier
    def gradient_boosting_classifier(train_x, train_y):
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=200)
        model.fit(train_x, train_y)
        return model

    @staticmethod
    # SVM Classifier
    def svm_classifier(train_x, train_y):
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', probability=True)
        model.fit(train_x, train_y)
        return model

    @staticmethod
    # SVM Classifier using cross validation
    def svm_cross_validation(train_x, train_y):
        from sklearn.grid_search import GridSearchCV
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', probability=True)
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
        grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
        grid_search.fit(train_x, train_y)
        best_parameters = grid_search.best_estimator_.get_params()
        for para, val in list(best_parameters.items()):
            print(para, val)
        model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
        model.fit(train_x, train_y)
        return model

    @staticmethod
    def mlp_classifier(train_x, train_y):
        # 多层线性回归 linear neural network
        from sklearn.neural_network import MLPRegressor
        # solver='lbfgs',  MLP的L-BFGS在小数据上表现较好，Adam较为鲁棒
        # SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）；SGD标识随机梯度下降。
        # alpha:L2的参数：MLP是可以支持正则化的，默认为L2，具体参数需要调整
        # hidden_layer_sizes=(5, 2) hidden层2层, 第一层5个神经元，第二层2个神经元)，2层隐藏层，也就有3层神经网络
        clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(train_x, train_y)
        return clf

    def read_data(self, data_file):
        data = pd.read_csv(data_file)
        train = data[:int(len(data) * 0.9)]
        test = data[int(len(data) * 0.9):]
        train_y = train.label
        train_x = train.drop('label', axis=1)
        test_y = test.label
        test_x = test.drop('label', axis=1)
        return train_x, train_y, test_x, test_y
