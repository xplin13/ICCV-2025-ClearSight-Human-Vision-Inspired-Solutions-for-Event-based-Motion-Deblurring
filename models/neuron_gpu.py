import torch
from spikingjelly.activation_based.neuron import  *

class CustomVinitLIFNode(BaseNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode ='m'):

        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.tau = tau
        self.register_memory('init_v', None)
        self.register_memory('init_v_flag', True)

    def v_float_to_tensor(self, x: torch.Tensor):
        if self.init_v_flag == True:
            assert self.init_v.shape == x.shape
            self.v = self.init_v
            self.init_v_flag = False

    def neuronal_charge(self, x: torch.Tensor):

        if self.v_reset is None:
            self.v = self.v + (x - self.v) / self.tau

        else:
            if isinstance(self.v_reset, float) and self.v_reset == 0.:
                self.v = self.v + (x - self.v) / self.tau
            else:  # charge equation
                self.v = self.v + (x - (self.v - self.v_reset)) / self.tau

    def forward(self, x: torch.Tensor, v_init: torch.Tensor = None):
        self.init_v = v_init[0]
        if self.step_mode == 's':
            return self.single_step_forward(x)
        elif self.step_mode == 'm':
            return self.multi_step_forward(x)
        else:
            raise ValueError(self.step_mode)
        
    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            v_seq.append(self.v)

        return torch.stack(y_seq), torch.stack(v_seq)

    def single_step_forward(self, x: torch.Tensor):
        
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike
