import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp


# Find jVMC package
import sys
sys.path.append(sys.path[0] + "../..")
import jVMC
import jVMC.global_defs as global_defs
from jVMC.nets.initializers import init_fn_args
from jVMC.util.symmetries import LatticeSymmetry

from typing import Union

from functools import partial


class RNNCellStack(nn.Module):
    """
    Implementation of a stack of RNN-cells which is scanned over an input sequence.
    This is achieved by stacking multiple 'vanilla' RNN-cells to obtain a deep RNN.
    Arguments:
        * ``hiddenSize``: size of the hidden state vector
        * ``actFun``: non-linear activation function
        * ``initScale``: factor by which the initial parameters are scaled
    Returns:
        New set of hidden states (one for each layer), as well as the last hidden state, that serves as input to the output layer
    """

    cells: list

    @ nn.compact
    def __call__(self, carry, newR):
        newCarry = jnp.zeros_like(carry)

        for j, (c, cell) in enumerate(zip(carry, self.cells)):
            current_carry, newR = cell(c, newR)
            newCarry = newCarry.at[j].set(current_carry)
        return newCarry, newR

# ** end class RNNCellStack


class RNN1DGeneral(nn.Module):
    """
    Implementation of an RNN which consists of an RNNCellStack with an additional output layer.
    This class defines how sequential input data is treated.
    Arguments:
        * ``L``: length of the spin chain
        * ``hiddenSize``: size of the hidden state vector
        * ``depth``: number of RNN-cells in the RNNCellStack
        * ``inputDim``: dimension of the input
        * ``actFun``: non-linear activation function
        * ``initScale``: factor by which the initial parameters are scaled
        * ``logProbFactor``: factor defining how output and associated sample probability are related. 0.5 for pure states and 1 for POVMs.
        * ``realValuedOutput``: Boolean indicating whether the network output is a real or complex number.
        * ``realValuedParams``: Boolean indicating whether the network parameters are real or complex parameters.
        * ``cell``: String ("RNN", "LSTM", or "GRU") or custom definition indicating which type of cell to use for hidden state  transformations.
    Returns:
        logarithmic wave-function coefficient or POVM-probability
    """

    L: int = 10
    hiddenSize: int = 10
    depth: int = 1
    inputDim: int = 2
    actFun: callable = nn.elu
    initScale: float = 1.0
    logProbFactor: float = 0.5
    realValuedOutput: bool = False
    realValuedParams: bool = True
    cell: Union[str, list] = "RNN"

    def setup(self):
        if isinstance(self.cell, str) and self.cell != "RNN":
            ValueError("Complex parameters for LSTM/GRU not yet implemented.")

        if self.realValuedParams:
            self.dtype = global_defs.tReal
            self.initFunction = jax.nn.initializers.variance_scaling(scale=self.initScale, mode="fan_avg", distribution="uniform")
        else:
            self.dtype = global_defs.tCpx
            self.initFunction = partial(jVMC.nets.initializers.cplx_variance_scaling, scale=self.initScale)

        if isinstance(self.cell, str):
            self.zero_carry = jnp.zeros((self.depth, 1, self.hiddenSize), dtype=self.dtype)
            if self.cell == "RNN":
                self.cells = [RNNCell(actFun=self.actFun, initFun=self.initFunction, dtype=self.dtype) for _ in range(self.depth)]
            elif self.cell == "LSTM":
                self.cells = [LSTMCell() for _ in range(self.depth)]
                self.zero_carry = jnp.zeros((self.depth, 2, self.hiddenSize), dtype=self.dtype)
            elif self.cell == "GRU":
                self.cells = [GRUCell() for _ in range(self.depth)]
            else:
                ValueError("Cell name not recognized.")
        else:
            self.cells = self.cell[0]
            self.zero_carry = self.cell[1]

        self.rnnCell = RNNCellStack(self.cells)
        init_args = init_fn_args(dtype=self.dtype, bias_init=jax.nn.initializers.zeros, kernel_init=self.initFunction)
        self.outputDense = nn.Dense(features=(self.inputDim-1) * (2 - self.realValuedOutput),
                                    use_bias=True, **init_args)

    def log_coeffs_to_log_probs(self, logCoeffs):
        phase = jnp.zeros((self.inputDim))
        if not self.realValuedOutput and self.realValuedParams:
            phase = 1.j*jnp.concatenate([jnp.array([0.0]), logCoeffs[self.inputDim-1:]]).transpose()
        amp = jnp.concatenate([jnp.array([0.0]), logCoeffs[:self.inputDim-1]]).transpose()

        return (self.logProbFactor * jax.nn.log_softmax(amp)).transpose() + phase 

    def __call__(self, x):
        _, probs = self.rnn_cell((self.zero_carry, jnp.zeros(self.inputDim)), jax.nn.one_hot(x, self.inputDim))

        return jnp.sum(probs)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell(self, carry, x):
        newCarry, out = self.rnnCell(carry[0], carry[1])
        logProb = self.log_coeffs_to_log_probs(self.outputDense(out))
        logProb = jnp.sum(logProb * x, axis=-1)
        return (newCarry, x), jnp.nan_to_num(logProb, nan=-35)

    def sample(self, batchSize, key):
        def generate_sample(key):
            myKeys = jax.random.split(key, self.L)
            _, sample = self.rnn_cell_sample(
                (self.zero_carry, jnp.zeros(self.inputDim)),
                (myKeys)
            )
            return sample[1]

        keys = jax.random.split(key, batchSize)
        return jax.vmap(generate_sample)(keys)

    @partial(nn.transforms.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def rnn_cell_sample(self, carry, x):
        newCarry, out = self.rnnCell(carry[0], carry[1])
        logCoeffs = self.log_coeffs_to_log_probs(self.outputDense(out))
        sampleOut = jax.random.categorical(x, jnp.real(logCoeffs) / self.logProbFactor)
        return (newCarry, jax.nn.one_hot(sampleOut, self.inputDim)), (jnp.nan_to_num(logCoeffs, nan=-35), sampleOut)


class GRUCell(nn.Module):
    @nn.compact
    def __call__(self, carry, state):
        current_carry, newR = nn.GRUCell(**init_fn_args(recurrent_kernel_init=jax.nn.initializers.orthogonal(dtype=global_defs.tReal)))(carry, state)
        return current_carry, newR[0]


class LSTMCell(nn.Module):
    @nn.compact
    def __call__(self, carry, state):
        current_carry, newR = nn.OptimizedLSTMCell(**init_fn_args(recurrent_kernel_init=jax.nn.initializers.orthogonal(dtype=global_defs.tReal)))(carry, state)
        return jnp.asarray(current_carry), newR


class RNNCell(nn.Module):
    """
    Implementation of a 'vanilla' RNN-cell, that is part of an RNNCellStack which is scanned over an input sequence.
    The RNNCell therefore receives two inputs, the hidden state (if it is in a deep part of the CellStack) or the 
    input (if it is the first cell of the CellStack) aswell as the hidden state of the previous RNN-cell.
    Both inputs are mapped to obtain a new hidden state, which is what the RNNCell implements.
    Arguments: 
        * ``initFun``: initialization function for parameters
        * ``actFun``: non-linear activation function
        * ``dtype``: data type of parameters
    Returns:
        new hidden state
    """

    initFun: callable = jax.nn.initializers.variance_scaling(scale=1e-1, mode="fan_avg", distribution="uniform")
    actFun: callable = nn.elu
    dtype: type = global_defs.tReal

    @nn.compact
    def __call__(self, carry, state):
        cellCarry = nn.Dense(features=carry.shape[-1],
                             use_bias=False,
                             **init_fn_args(dtype=self.dtype,
                                            bias_init=jax.nn.initializers.zeros,
                                            kernel_init=self.initFun))
        cellState = nn.Dense(features=carry.shape[-1],
                             use_bias=False,
                             **init_fn_args(dtype=self.dtype,
                                            bias_init=jax.nn.initializers.zeros,
                                            kernel_init=self.initFun))

        newCarry = self.actFun(cellCarry(carry[0]) + cellState(state))[None, :]
        
        return newCarry, newCarry[0]



class RNN1DGeneralSym(nn.Module):
    """
    Implementation of an RNN which consists of an RNNCellStack with an additional output layer.
    It uses the RNN class to compute probabilities and averages the outputs over all symmetry-invariant configurations.

    Arguments: 
        * ``orbit``: collection of maps that define symmetries (instance of ``util.symmetries.LatticeSymmetry``)
        * ``L``: length of the spin chain
        * ``hiddenSize``: size of the hidden state vector
        * ``depth``: number of RNN-cells in the RNNCellStack
        * ``inputDim``: dimension of the input
        * ``actFun``: non-linear activation function
        * ``initScale``: factor by which the initial parameters are scaled
        * ``logProbFactor``: factor defining how output and associated sample probability are related. 0.5 for pure states and 1 for POVMs.
        * ``z2sym``: for pure states; implement Z2 symmetry

    Returns:
        Symmetry-averaged logarithmic wave-function coefficient or POVM-probability
    """
    orbit: LatticeSymmetry
    L: int = 10
    hiddenSize: int = 10
    depth: int = 1
    inputDim: int = 2
    actFun: callable = nn.elu
    initScale: float = 1.0
    logProbFactor: float = 0.5
    realValuedOutput: bool = False
    realValuedParams: bool = True
    cell: Union[str, list] = "RNN"
    z2sym: bool = False

    def setup(self):

        self.rnn = RNN1DGeneral(L=self.L, hiddenSize=self.hiddenSize, depth=self.depth,
                                inputDim=self.inputDim,
                                actFun=self.actFun, initScale=self.initScale,
                                logProbFactor=self.logProbFactor,
                                realValuedOutput=self.realValuedOutput,
                                realValuedParams=self.realValuedParams)


    def __call__(self, x):

        x = jax.vmap(lambda o, s: jnp.dot(o, s), in_axes=(0, None))(self.orbit.orbit, x)

        def evaluate(x):
            return self.rnn(x)

        res = jnp.mean(jnp.exp((1. / self.logProbFactor) * jax.vmap(evaluate)(x)), axis=0)

        if self.z2sym:
            res = 0.5 * (res + jnp.mean(jnp.exp((1. / logProbFactor) * jax.vmap(evaluate)(1 - x)), axis=0))

        logProbs = self.logProbFactor * jnp.log(res)

        return logProbs

    def sample(self, batchSize, key):

        key1, key2 = jax.random.split(key)

        configs = self.rnn.sample(batchSize, key1)

        orbitIdx = jax.random.choice(key2, self.orbit.orbit.shape[0], shape=(batchSize,))

        configs = jax.vmap(lambda k, o, s: jnp.dot(o[k], s), in_axes=(0, None, 0))(orbitIdx, self.orbit.orbit, configs)

        if self.z2sym:
            key3, _ = jax.random.split(key2)
            flipChoice = jax.random.choice(key3, 2, shape=(batchSize,))
            configs = jax.vmap(lambda b, c: jax.lax.cond(b == 1, lambda x: 1 - x, lambda x: x, c), in_axes=(0, 0))(flipChoice, configs)

        return configs

# ** end class RNN1DGeneralSym
