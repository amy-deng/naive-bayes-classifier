# coding: utf-8
from NaiveBayesClassifier.unknownException import Unknown

import operator


class Classifier(object):
    """
    Naive Bayes classifier
    """

    def __init__(self, trained_data):
        self.data = trained_data
        self.default_prob = 0.000001

    def classify(self, attr_dict):
        classes = self.data.getClasses()

        probs_of_classes = {}

        for class_name in classes:

            # P(Feature_1|Class_i)
            features_prob_list = [self.getFeatureProb(attr_dict[attr], attr, class_name) for attr in attr_dict]

            # P(Feature_1|Class_i) * P(Feature_2|Class_i) * ... * P(Feature_n|Class_i)
            try:
                feature_set_prob = reduce(lambda a, b: a * b, (i for i in features_prob_list if i))
            except:
                feature_set_prob = 0

            probs_of_classes[class_name] = feature_set_prob * self.getPrior(class_name)

        return sorted(probs_of_classes.items(),
                      key=operator.itemgetter(1),
                      reverse=True)

    def getPrior(self, class_name):
        return self.data.getClassCount(class_name) * 1.0 / self.data.getCount()

    def getFeatureProb(self, feature, attribute, class_name):
        # p(feature|Class_i) = p(Class_i|feature) * p(feature) / p(Class_i)
        class_count = self.data.getClassCount(class_name)

        # Return None and not to calculate it if the feature is unknown.
        try:
            feature_frequency = self.data.getFrequency(feature, attribute, class_name)
        except Unknown as e:
            return None

        # the feature is not seen in this class but others.
        if feature_frequency is None:
            return self.default_prob

        return feature_frequency * 1.0 / class_count
