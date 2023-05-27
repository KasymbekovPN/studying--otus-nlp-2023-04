from src.hw_002_py_tourch.source.args.args import Args


class FloatArgSequenceSource:
    def __init__(self,
                 args: 'Args'):
        self._args_source = args

    def __call__(self, *args, **kwargs):
        # kwargs['length']
        pass
