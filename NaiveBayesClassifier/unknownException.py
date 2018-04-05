# coding: utf-8


class Unknown(Exception):
    """
    Handle exceptions
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Unknown feature '{}' in the training set.".format(self.value)
