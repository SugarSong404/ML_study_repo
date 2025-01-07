import pickle
import numpy as np

class DecisionTree:
    def __init__(self, data, label, task='classification', reg = None):
        if data is not None:
            self.tree = None
            self.task = task
            if task=='regression': 
                reg = np.ones(data.shape[1])
            self.label = label
            self.feature_arr = [{'name': i, 'reg': reg[i]} for i in range(data.shape[1])]
            self.data = self.__prepare_data(data)

    def __prepare_data(self, data):
        if self.task=='regression': return data
        new_data = np.empty(data.shape, dtype=object)
        for i in range(data.shape[1]):
            if self.feature_arr[i]['reg'] == 1:
                new_data[:, i] = data[:, i].astype('float')
            else:
                new_data[:, i] = data[:, i]
        return new_data

    def train(self):
        self.tree = self.__createNode(self.data, self.label.flatten(), self.feature_arr)

    def predict(self, data):
        predictions = [self.__predict_single(sample, self.tree) for sample in data]
        return np.array(predictions)

    def __predict_single(self, ori_sample, tree):
        sample = self.__prepare_data(np.array([ori_sample]))[0]
        if not isinstance(tree, dict):
            return tree
        feature = list(tree.keys())[0]
        value = sample[feature]
        for key in tree[feature]:
            if "<=" in key:
                threshold = float(key.split("<=")[-1])
                if value <= threshold:
                    return self.__predict_single(sample, tree[feature][key])
            elif ">" in key:
                threshold = float(key.split(">")[-1])
                if value > threshold:
                    return self.__predict_single(sample, tree[feature][key])
            elif "=" in key:
                threshold = key.split("=")[-1]
                if value == threshold:
                    return self.__predict_single(sample, tree[feature][key])
        return None

    def __createNode(self, data, label, feature_arr):
        labelList = label.flatten()
        if self.task == 'classification' and np.all(labelList == labelList[0]):
            return labelList[0]
        if self.task == 'regression' and len(labelList) <= 1:
            return np.mean(labelList)
        if data.shape[1] == 0:
            return self.__majorCnt(labelList) if self.task == 'classification' else np.mean(labelList)

        best_id, splitVal = self.__find_bestFeature(data, label, feature_arr)
        node = {feature_arr[best_id]['name']: {}}

        if splitVal is None:
            featureVals = set(data[:, best_id])
            for val in featureVals:
                mask = data[:, best_id] == val
                subData = np.delete(data[mask], best_id, axis=1)
                subLabel = label[mask]
                subFeatureNames = feature_arr[:best_id] + feature_arr[best_id+1:]
                node[feature_arr[best_id]['name']][f"={val}"] = self.__createNode(subData, subLabel, subFeatureNames)
        else:
            mask = data[:, best_id] <= splitVal
            subDataLeft = np.delete(data[mask], best_id, axis=1)
            subLabelLeft = label[mask]
            subDataRight = np.delete(data[~mask], best_id, axis=1)
            subLabelRight = label[~mask]
            subFeatureNames = feature_arr[:best_id] + feature_arr[best_id+1:]
            node[feature_arr[best_id]['name']][f"<=%.3f" % splitVal] = self.__createNode(subDataLeft, subLabelLeft, subFeatureNames)
            node[feature_arr[best_id]['name']][f">%.3f" % splitVal] = self.__createNode(subDataRight, subLabelRight, subFeatureNames)

        return node

    def __find_bestFeature(self, data, label, feature_arr):
        bestScore = float('inf')
        best_id = -1
        bestSplitVal = None
        for i in range(data.shape[1]):
            if feature_arr[i]['reg'] == 1 or self.task == 'regression':
                sortedIndices = np.argsort(data[:, i])
                sortedData, sortedLabel = data[sortedIndices], label[sortedIndices]
                for j in range(1, len(sortedData)):
                    if sortedLabel[j] != sortedLabel[j - 1]:
                        splitVal = (sortedData[j, i] + sortedData[j - 1, i]) / 2
                        score = self.__calculate_split_score(sortedData[:, i], sortedLabel, splitVal)
                        if score < bestScore:
                            bestScore = score
                            best_id = i
                            bestSplitVal = splitVal
            else:
                score = self.__calculate_split_score(data[:, i], label, None)
                if score < bestScore:
                    bestScore = score
                    best_id = i
                    bestSplitVal = None
        return best_id, bestSplitVal

    def __calculate_split_score(self, featureVals, label, splitVal):
        if self.task == 'classification':
            return self.__calculate_gini_split(featureVals, label, splitVal)
        elif self.task == 'regression':
            return self.__calculate_mse_split(featureVals, label, splitVal)

    def __calculate_gini_split(self, featureVals, label, splitVal):
        if splitVal is None:
            uniqueVals = np.unique(featureVals)
            gini = 0
            for val in uniqueVals:
                mask = featureVals == val
                gini += (np.sum(mask) / len(featureVals)) * self.__calculate_gini(label[mask])
        else:
            mask = featureVals <= splitVal
            leftGini = self.__calculate_gini(label[mask])
            rightGini = self.__calculate_gini(label[~mask])
            gini = (np.sum(mask) / len(featureVals)) * leftGini + (np.sum(~mask) / len(featureVals)) * rightGini
        return gini

    def __calculate_gini(self, label):
        _, counts = np.unique(label, return_counts=True)
        probs = counts / len(label)
        return 1 - np.sum(probs ** 2)

    def __calculate_mse_split(self, featureVals, label, splitVal):
        mask = featureVals <= splitVal
        leftMSE = self.__calculate_mse(label[mask])
        rightMSE = self.__calculate_mse(label[~mask])
        mse = (np.sum(mask) / len(featureVals)) * leftMSE + (np.sum(~mask) / len(featureVals)) * rightMSE
        return mse

    def __calculate_mse(self, label):
        if len(label) == 0:return 0
        mean_value = np.mean(label)
        return np.mean((label - mean_value) ** 2)

    def __majorCnt(self, labelList):
        unique, counts = np.unique(labelList, return_counts=True)
        return unique[np.argmax(counts)]

    def score(self, data, label):
        predictions = self.predict(data)
        if self.task == 'classification':
            accuracy = np.mean(predictions == label.ravel())
            return accuracy
        elif self.task == 'regression':
            total_var = np.sum((label.ravel() - np.mean(label)) ** 2)
            residual_var = np.sum((predictions - label.ravel()) ** 2)
            return 1 - (residual_var / total_var)

    def show(self):
        tree = self.tree
        def traverse_tree(tree, depth=0):
            indent = "  " * depth
            if isinstance(tree, dict):
                for feature, branches in tree.items():
                    print(f"{indent}Feature: {feature}")
                    for value, subtree in branches.items():
                        print(f"{indent}  └─ Value: {value}")
                        traverse_tree(subtree, depth + 2)
            else:
                print(f"{indent}  └─ Label: {tree}")

        print("Decision Tree Structure:")
        traverse_tree(tree)

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'tree': self.tree, 'feat': self.feature_arr ,'task':self.task}, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
            self.tree = loaded['tree']
            self.task = loaded['task']
            self.feature_arr = loaded['feat']
