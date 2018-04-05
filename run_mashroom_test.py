# coding: utf-8

from NaiveBayesClassifier.trainedData import TrainedData
from NaiveBayesClassifier.classifier import Classifier
import numpy


class Trainer(object):
    def __init__(self):
        self.data = TrainedData()

    def contentFormat(self, attributes):

        attribute_name_list = [
            'cap-shape',
            'cap-surface',
            'cap-color',
            'bruises',
            'odor',
            'gill-attachment',
            'gill-spacing',
            'gill-size',
            'gill-color',
            'stalk-shape',
            'stalk-root',
            'stalk-surface-above-ring',
            'stalk-surface-below-ring',
            'stalk-color-above-ring',
            'stalk-color-below-ring',
            'veil-type',
            'veil-color',
            'ring-number',
            'ring-type',
            'spore-print-color',
            'population',
            'habitat',
        ]
        attribute_list = attributes.split(',')  # 22 attributes
        attr_dict = {}
        for i in range(len(attribute_list)):
            attr_dict[attribute_name_list[i]] = attribute_list[i]

        return attr_dict

    def train(self, attributes, class_came):

        self.data.addClass(class_came)
        attribute_dict = self.contentFormat(attributes)
        for attribute in attribute_dict:
            # Missing Attribute Values: 2480 of them (denoted by "?"), all for attribute #11 'stalk-root'.
            if attribute_dict[attribute] != '?':
                self.data.addFeature(attribute_dict[attribute], attribute, class_came)


if __name__ == '__main__':
    trainer = Trainer()

    file_path = 'dataset/mushroom/agaricus-lepiota.data'
    training_set = []
    testing_set = []
    for line in open(file_path):
        line = line.strip('\n')
        split_data = line.split(',', 1)

        if numpy.random.ranf() < 0.25:
            testing_set.append(split_data)
        else:
            training_set.append(split_data)

    for tr_s in training_set:
        trainer.train(tr_s[1], tr_s[0])
    print 'frequencies', trainer.data.frequencies

    classifier = Classifier(trainer.data)

    num_accurate = 0

    for te_s in testing_set:
        classification = classifier.classify(trainer.contentFormat(te_s[1]))
        if classification[0][0] == te_s[0]:
            num_accurate += 1
            # print classification
    print '========result========='
    print 'num_training:', len(training_set), 'num_testing:', len(testing_set), 'num_accurate:', num_accurate
    print 'accuracy:', num_accurate * 1.0 / len(testing_set)
