from ATensor import ATensor

class AModule:
    def __init__(self):
        self.parameters = {}
        self.gradients = {}

    def __call__(self, x: ATensor) -> ATensor:
        return forward(x)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError