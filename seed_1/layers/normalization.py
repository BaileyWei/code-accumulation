import torch
import torch.nn as nn


class ConditionalLayerNormalization(nn.Module):
    def __init__(self, input_dim, cond_dim):
        super(ConditionalLayerNormalization, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """

        self.epsilon = 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        self.beta = nn.Linear(in_features=self.cond_dim, out_features=input_dim)
        self.gamma = nn.Linear(in_features=self.cond_dim, out_features=input_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # 下面这两个为什么都初始化为0呢?
        # 为了防止扰乱原来的预训练权重，两个变换矩阵可以全零初始化
        # （单层神经网络可以用全零初始化，连续的多层神经网络才不应当用全零初始化），
        # 这样在初始状态，模型依然保持跟原来的预训练模型一致。
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)

    def forward(self, input: torch.FloatTensor, cond: torch.FloatTensor) -> torch.FloatTensor:
        """
            如果是条件Layer Norm，则cond不是None
        """
        for _ in range(len(input.shape) - len(cond.shape)):
            cond = cond.unsqueeze(1)

        beta = self.beta(cond)
        gamma = self.gamma(cond)

        output = input
        mean = torch.mean(output, dim=-1).unsqueeze(-1)
        output = output - mean

        variance = torch.mean(output ** 2, dim=-1).unsqueeze(-1)
        std = (variance + self.epsilon) ** 0.5
        output = output / std

        output = output * gamma
        output = output + beta

        return output  # F.layer_norm(input, tuple((self.input_dim,)), gamma, beta, self.epsilon)

