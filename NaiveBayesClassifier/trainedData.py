# coding: utf-8
from NaiveBayesClassifier.unknownException import Unknown


class TrainedData(object):

    """
    Process attributes and classes
    Calculate the probability
    """

    def __init__(self):
        self.count_of_classes = {}
        self.frequencies = {}

    def addClass(self, class_name):
        self.count_of_classes[class_name] = self.count_of_classes.get(class_name, 0) + 1

    def addFeature(self, feature, attribute_name, class_name):
        if not attribute_name in self.frequencies:
            self.frequencies[attribute_name] = {}

        if not feature in self.frequencies[attribute_name]:
            self.frequencies[attribute_name][feature] = {}

        self.frequencies[attribute_name][feature][class_name] = self.frequencies[attribute_name][feature].get(
            class_name, 0) + 1

    def getCount(self):
        return sum(self.count_of_classes.values())

    def getClasses(self):
        return self.count_of_classes.keys()

    def getClassCount(self, class_name):
        return self.count_of_classes.get(class_name, None)

    def getFrequency(self, feature, attribute, class_name):
        if feature in self.frequencies[attribute]:
            foundFeature = self.frequencies[attribute][feature]
            return foundFeature.get(class_name)
        else:
            raise Unknown(feature)
