class BaseLayer:
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def compile(self, *args, **kwargs):
        raise NotImplementedError()
