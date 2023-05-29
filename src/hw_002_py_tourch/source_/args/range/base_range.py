
# todo del
class BaseRange:
    TYPE_SETTER_EXC_MSG = 'Type setter is unsupported'

    def __init__(self, type_):
        self._type = type_

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        raise Exception(self.TYPE_SETTER_EXC_MSG)
