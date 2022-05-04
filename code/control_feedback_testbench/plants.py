# Mihir Savadi 4th April 2022

from multiprocessing.sharedctypes import Value
from tkinter import N
from turtle import forward

from numpy import require
from lib import *
from math import ceil

from .base import plantBase

class FOPDT(plantBase) :
    """First Order plus Dead time controller. See https://apmonitor.com/pdc/index.php/Main/FirstOrderSystems and 
        https://apmonitor.com/pdc/index.php/Main/FirstOrderOptimization

        Parent : plantBase
    """
    
    def __init__(self, params={}) :
        """Constructor for FOPDT plant.

        Parameters
        ----------
        params : dict
            key: "K"     -> K constant for FOPDT calculation.
            key: "tau"   -> tau constant for FOPDT calculation.
            key: "theta" -> theta constant for FOPDT calculation.
        """
        super().__init__(params)

        for arg in ['K', 'tau', 'theta'] :
            if arg not in params :
                raise ValueError(f"FOPDT class initializer missing an {arg} key in its 'param' dict argument.")

        self._K = params["K"]
        self._tau = params["tau"]
        self._theta = params["theta"]

        self._u_memory = [0.0]*ceil(self._theta+1)  # records the history of u up until (t-theta)
        self._y_memory = [0.0]*2                    # records only last 2 y values to get gradient

    def execute(self, u: float) -> float :
        """Executes for one timestep the output of the FOPDT model. Uses second function described here 
            https://apmonitor.com/pdc/index.php/Main/FirstOrderOptimization

        Parameters
        ----------
        inputs : Dict[str, float]
            the only entry in this dict should be a single float with key 'u'

        Returns
        -------
        float
            float representing 'y' output of controller
        """

        output = exp(-1/self._tau) * (self._y_memory[-1] - 0) +\
                    (1 - exp(-1/self._tau)) * self._K * (self._u_memory[0] - 0) + 0

        self._u_memory.pop(0)
        self._u_memory.append(u)
        self._y_memory.pop(0)
        self._y_memory.append(output)

        return output

    def getAuxiliaryInfo(self) -> Dict :
        """Returns a dictionary of all the private class fields.
        """
        return {
            'K'     : self._K,
            'tau'   : self._tau,
            'theta' : self._theta,
            'u'     : self._u_memory,
            'y'     : self._y_memory,
        }

class SOPDT(plantBase) :
    """Second Order Plus Dead Time plant.
        https://apmonitor.com/pdc/index.php/Main/SecondOrderSystems

        Parent : plantBase 
        TODO
    """
    pass

class DNN(plantBase) :
    """A plant based on a deep neural network that is used to model an existing black-box of a plant. Its utility is
       that it models said black box of a plant whilst providing a known function through which a path for 
       backpropagation (for training of another neural net) can be done. The input is an n-dimensional tensor 
       ({r[t], r[t-1], r[t-2],...,r[t-n]}), output is a scalar, and number of hidden layers as well as their sizes are 
       left as variables. Primarily designed to be used in the RTCFL circuit, but general enough to be used elsewhere.

       Note that this was tested to perform like garbage in the RTCFL.

       Parent : plantBase
    """

    def __init__(self, name: str = "plant_nn", params: dict=None):
        """Constructor for DNN plant.

        Parameters
        ----------
        name : str, optional
            By default "plant_nn". Just for printing and CLI interface purposes.
        params : dict
            key: "n" -> int : determines the 1D input tensor size, which takes in up to [t-n] r value's as inputs,  
                allowing for 'memory' of past inputs.
            key: "hidden layers" -> list : a list of integers that correspond to size of each hidden layer. Must have at  
            least 1 element. 
            key: "activation function" -> torch.nn : Must be a valid pytorch activation function, like torch.nn.ReLU, 
            torch.nn.Sigmoid, torch.nn.tanh, or torch.nn.LeakyReLU.
            key: "learning rate" -> must be a float > 0
            key: "device" -> str : must be either 'cuda' or 'cpu'
        """
        # perform checks to see if arguments are sound
        for arg in ['n', 'hidden layers', 'activation function', 'learning rate', 'device'] :
            if arg not in params :
                raise ValueError(f"plant.DNN class initializer missing an {arg} key in its 'param' dict argument.")
        
        if type(params['n']) != int :
            raise ValueError(f"plant.dnn class initializer requires value for 'n' key in 'param' dict argument to be an int. Right now it is a {type(params['n'])}='{params['n']}'.")

        if type(params['hidden layers']) != list :
            raise ValueError(f"plant.dnn class initializer requires value for 'hidden layers' key in 'param' dict argument to be a list of ints. Right now it is a {type(params['hidden layers'])}='{params['hidden layers']}'.")
        if len(params['hidden layers']) < 1 :
            raise ValueError(f"plant.dnn class initializer requires len for 'hidden layers' key in 'param' dict argument to be at least 1. Right now it is of len='{len(params['hidden layers'])}'.")


        for i, element in enumerate(params['hidden layers']) :
            if type(element) != int :
                raise ValueError(f"plant.dnn class initializer requires value for 'hidden layers' key in 'param' dict argument to be a list of ints. Right now element {i} in said list is a {type(params['hidden layers'][i])}='{params['hidden layers'][i]}'.")

        if type(params['learning rate']) != float :
            raise ValueError(f"plant.dnn class initializer requires value for 'learning rate' key in 'param' dict argument to be of float type. Right now it is a '{type(params['learning rate'])}'.")
        if params['learning rate'] <= 0 :
            raise ValueError(f"plant.dnn class initializer requires value for 'learning rate' key in 'param' dict argument to be greater than 0. Right now it is a '{params['learning rate']}'.")

        valid_devices = ['cuda', 'cpu']
        if type(params['device']) != str or params['device'] not in valid_devices :
            raise ValueError(f"plant.dnn class initializer requires value for 'device' key in 'param' dict argument to be in {valid_devices}. Right now it is a {type(params['device'])}='{params['device']}'.")

        # create internal fields and the pytorch DNN model class (and an instance of it) accordingly
        class NN(torch.nn.Module) :
            def __init__(self):
                super().__init__()
                # use nn.ModuleList() instead of a simple [] list so that pytorch can index and find the layers.
                self._layerList = torch.nn.ModuleList()
                self._layerList.append(torch.nn.Linear(params['n'], params['hidden layers'][0]))
                self._layerList.append(params['activation function']())
                for i in range(0, len(params['hidden layers'])-1) :
                    self._layerList.append(torch.nn.Linear(params['hidden layers'][i], params['hidden layers'][i+1]))
                    self._layerList.append(params['activation function']())
                self._layerList.append(torch.nn.Linear(params['hidden layers'][-1], 1))
                self._layerList.append(params['activation function']())

                self._inputMemory = torch.FloatTensor([0.0]*params['n']).to(device=params['device'])

            def forward(self, r) :
                """Note that the 'r' input here must be a scalar. The input memory vector is handled internally.
                """
                self._inputMemory = torch.cat((torch.FloatTensor([r]).to(device=params['device']), 
                                               self._inputMemory[:-1]))
                
                output = self._layerList[0](self._inputMemory)
                for i in range(1, len(self._layerList)) :
                    output = self._layerList[i](output)
                return output

            def getMemory(self) -> torch.FloatTensor :
                """Returns the currently maintained input memory vector.
                """
                return self._inputMemory

        # establish all internal fields
        self._instanceName       = name
        self._inputLength        = params['n']
        self._hiddenLayersLayout = params['hidden layers']
        self._activationFunction = params['activation function']
        self._learningRate       = params['learning rate']
        self._device             = params['device']

        self._model              = NN().to(self._device)
        self._lossFunction       = torch.nn.MSELoss()
        self._optimizer          = torch.optim.AdamW(self._model.parameters(), lr=self._learningRate)

    def execute(self, r) -> torch.FloatTensor :
        """Passes an input into the controller and returns an output, so straightforward inference. Sets the 
        internal NN model to eval mode. Returns a tensor of the output.

        Parameters
        ----------
        x : int or float
            must be a scalar value. represents 'u'.

        Returns
        -------
        torch.FloatTensor
            output of DNN
        """
        # set the pytorch model module into evaluation mode (disables any dropout layers) and alters normalization
        # characteristics
        self._model.eval()
        return self._model(r)

    def train(self, 
              plant_nn_output : torch.FloatTensor, 
              plant_bb_output : torch.FloatTensor,
              printInfo       : bool = False 
              ) -> torch.FloatTensor :
        """Does one round of forward-backward pass and weight update. Returns the loss.

        Parameters
        ----------
        plant_nn_output : torch.FloatTensor
            A tensor containing recorded plant_nn outputs. Note that each element in this tensor needs to be a tensor
            itself - each element represents one output that the plant_nn produced.
        plant_bb_output : torch.FloatTensor
            A tensor containing recorded plant_bb outputs associated to corresponding plant_nn output in the 
            plant_nn_output tensor. This is used to calculate average loss.
        printInfo : bool, optional
            If true training data information (loss etc) is printed to CLI, otherwise silenced.

        Returns
        -------
        torch.FloatTensor
            loss value
        """
        # set the model into train mode (enables any dropout layers) and alters normalization characteristics.
        self._model.train()
        # zero the gradients. needed cos pytorch accumulates gradients from previous passes.
        self._optimizer.zero_grad()
        # compute the loss after computing forward pass with training input and running it with label through loss
        # function.
        loss = self._lossFunction(plant_nn_output.squeeze(), plant_bb_output)
        loss.requires_grad = True
        # if print enabled then print.
        if printInfo :
            print(f"\t{self._instanceName} Training loss: {loss}.")
        # backward pass. 
        loss.backward() # compute the gradients
        self._optimizer.step() # update the weights accordingly

        return loss

    def getAuxiliaryInfo(self) :
        """Returns a dictionary of all the private class fields.
        """
        return {
            'input vector'        : self._inputLength,
            'hidden layers'       : self._hiddenLayersLayout,
            'activation function' : self._activationFunction,
            'learning rate'       : self._learningRate,
            'device'              : self._device,
            'model'               : self._model,
            'model input memory'  : self._model.getMemory(),
            'loss function'       : self._lossFunction
        }


class RNN(plantBase) :
    """A RNN based plant that is used to model an existing black box of a plant. Its utility is that it models said
       black box of a plant whilst providing a known function through which a path for backpropagation (for training of
       another neural net) can be done. 

       Since plants have a scalar input and output, so does this NN. The number of RNN units to stack one after another
       is variable. Each RNN unit, i, has 1 scalar output: out_i[t], and 2 scalar inputs: in_i[t] and out_i[t-1]. Hidden 
       states are initialized to 0.

       Training is done by pushing a batch of 'r' sequences through the model. A batch can have sequences of any length, 
       but long consistently sized sequences are preferred. For each 'r' value in a sequence, the loss is calculated by
       getting the squared difference between a black box plant and the output of the RNN. This loss of each 'r' value
       is summed, and that is the loss for the given sequence. The losses for all the sequences in the batch are then
       averaged, from which the final gradients are calculated.

       Parent : plantBase
    """

    def __init__(self, name: str = "plant_nn", stackCount: int = 1, learningRate: float=0.01, device: str='cpu') :
        """Constructor for RNN plant.

        Parameters
        ----------
        name : str, optional
            By default "plant_nn". Just for printing and CLI interface purposes.
        stackCount : int, optional
            The number of RNN units to stack one after another, by default 1. Must be >= 1.
        learningRate : int, optional
            Must be > 0.
        device : str, optional
            Must either be 'cpu' or 'gpu'.
        """
        # perform argument checks.
        if stackCount < 1 :
            raise ValueError(f"plant.RNN class initializer needs 'stackCount' argument to be >= 1. Now it is '{stackCount}'.")
        if learningRate <= 0 :
            raise ValueError(f"plant.RNN class initializer needs 'learningRate' argument to be > 0. Now it is '{learningRate}'.")

        # create RNN model class.
        class RNN(torch.nn.Module) :
            def __init__(self):
                super().__init__()

                self._LinearLayer = torch.nn.Linear(2, 1)
                self._ReLULayer   = torch.nn.ReLU()
                self._hiddenStateList = []
                for i in range(stackCount) :
                    self._hiddenStateList.append(torch.FloatTensor([0]).to(device=device))

            def forward(self, r: int) :
                """
                Parameters
                ----------
                r : int
                    'r' must be scalar
                """
                output = self._ReLULayer(self._LinearLayer(torch.cat((torch.FloatTensor([r]).to(device=device), 
                                                                      self._hiddenStateList[0]))))
                self._hiddenStateList[0] = output
                for i in range(1, len(self._hiddenStateList)) :
                    output = self._ReLULayer(self._LinearLayer(torch.cat((output, self._hiddenStateList[i]))))
                    self._hiddenStateList[i] = output
                return output

            def peekHiddenStates(self) -> list :
                return self._hiddenStateList

            def detachHiddenLayers(self) :
                for hiddenlayer in self._hiddenStateList :
                    hiddenlayer.detach_()

        # establish all internal fields
        self._stackCount   = stackCount
        self._learningRate = learningRate
        self._device       = device
        self._model        = RNN().to(self._device)
        self._lossFunction = torch.nn.MSELoss()
        self._optimizer    = torch.optim.AdamW(self._model.parameters(), lr=self._learningRate)

    def execute(self, r: float) -> torch.FloatTensor :
        """Straightforward inference.

        Parameters
        ----------
        r : float
            Input

        Returns
        -------
        torch.FloatTensor
            Output, 1-D vector with one element.
        """
        self._model.eval()
        return self._model(r)

    def train(self, r_sequences: list, plant_bb_function: plantBase.execute) -> torch.FloatTensor :
        """Training is done by pushing a batch of 'r' sequences through the model. A batch can have sequences of any length, 
       but long consistently sized sequences are preferred. For each 'r' value in a sequence, the loss is calculated by
       getting the squared difference between a black box plant and the output of the RNN. This loss of each 'r' value
       is summed, and that is the loss for the given sequence. The losses for all the sequences in the batch are then
       averaged, from which the final gradients are calculated.

        Parameters
        ----------
        r_sequences : list
            the list, or batch, of 'r' sequences.
        plant_bb_function : plantBase.execute
            the function pointer used to calculate the black box plant outputs from which to calculate loss. Must only 
            take 1 argument, that is the input.

        Returns
        -------
        torch.FloatTensor
            loss value
        """
        self._model.train()
        # zero the gradients. needed cos pytorch accumulates gradients from previous passes.
        self._optimizer.zero_grad()
        self._model.detachHiddenLayers()
        # compute loss
        loss_sum = torch.tensor(0, dtype=float, requires_grad=True)
        for sequence in r_sequences :
            lossList = torch.tensor([])
            for r in sequence :
                newloss = (self._model(r) - plant_bb_function(r))**2
                lossList = torch.cat((lossList, newloss))
            loss_sum = torch.sum(lossList)
        loss = loss_sum / len(r_sequences)
        # backward pass
        loss.backward(retain_graph=True) # compute the gradients
        self._optimizer.step() # update the weights accordingly

        return loss

    def getAuxiliaryInfo(self) :
        """Returns a dictionary of all the private class fields.
        """
        return {
            'stack count'         : self._stackCount,
            'learning rate'       : self._learningRate,
            'device'              : self._device,
            'model'               : self._model,
            'model hidden states' : self._model.peekHiddenStates(),
            'loss function'       : self._lossFunction,
            'optimizer'           : self._optimizer
        }