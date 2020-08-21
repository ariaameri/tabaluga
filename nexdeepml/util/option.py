from abc import abstractmethod


class Option:

    def map(self, function):
        if type(self) is Some:
            return Some(function(self.value))
        else:
            return self

    def get_or_else(self, default_value):
        if type(self) is Some:
            return self.value
        else:
            return default_value

    def filter(self, function):
        if type(self) is Some:
            return self if function(self.value) else Nothing()
        else:
            return self

    @abstractmethod
    def get(self):

        pass


class Some(Option):

    def __init__(self, value):

        self.value = value

    def get(self):

        return self.value


class Nothing(Option):

    def get(self):

        raise AttributeError

