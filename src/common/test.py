
def repeat(times):
    def wrap(f):
        def call(*args):
            for i in range(0, times):
                f(*args)

        return call
    return wrap
