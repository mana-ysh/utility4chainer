
import numpy as np


class ProximalOptimizer(object):
    def __init__(self, opt):
        """
        Optimization via proximal gradient methods,
        which is useful for producing sparse solutions or something.

        Args
        - opt : standard gradient optimizer instance, such as SGD or Adam
        """
        self.opt = opt
        self.trg_pname = set()

    def _proximal_step(self):
        for (name, param) in self.opt.target.namedparams():
            if name in self.trg_pname:
                self._proximal_update(param)


    def _priximal_update(self, param):
        """
        update formulation :
        W^{new} = argmin r(Θ) + || W - W' ||
        """
        raise NotImplementedError()

    def _inner_setup(self, model):
        # assuming all kinds of parameter are registered
        for (name, param) in model.namedparams():
            self.trg_pname.add(name)

    def setup(self, model):
        self.opt.setup(model)
        self._inner_setup(model)

    def update(self):
        self.opt.update()
        self._proximal_step()


class L1ProximalOptimizer(ProximalOptimizer):
    def __init__(self, opt, strength):
        super(L1ProximalOptimizer, self).__init__(opt)
        self.strength = strength

    def _proximal_update(self, W):
        W.data = np.multiply(np.sign(W.data),
                             np.maximum(np.abs(W.data) - self.strength, 0))
