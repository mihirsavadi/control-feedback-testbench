# Mihir Savadi 4 April 2022

from lib import *
from .base import controllerBase, plantBase

memorySizeMin = 5 # general variable to set memory size variable. Never make below 1.

class PID_static(controllerBase) :
    """Regular old static PID controller. P, I, and D parameters must be set in the constructor, but can also be 
        set otherwise.

    Parent : controllerBase
    """
    
    def __init__(self, params) :
        """Constructor for Static PID plant.

        Parameters
        ----------
        params : dict
            key: "P" -> P constant for PID calculation.
            key: "I" -> P constant for PID calculation.
            key: "D" -> P constant for PID calculation.

        """
        super().__init__(params)

        for arg in ['P', 'I', 'D'] :
            if arg not in params :
                raise ValueError(f"PID_static class initializer missing an {arg} key in its 'param' dict argument.")

        self._P = params["P"]
        self._I = params["I"]
        self._D = params["D"]

        self._errorSum  = 0
        self._lastError = 0

        # Below are the last calculated potential, integral and derivative values, intialized to 0. 
        self._potential  = 0
        self._integral   = 0
        self._derivative = 0

    def execute(self, r, y) -> float :
        """Gets output of static PID controller given inputs.

        Parameters
        ----------
        inputs : Dict[float]
            Dictionary with the following keys and description
            'r' : the current r_input
            'y' : the last y value

        Returns
        -------
        float
            the output of the full PID calculation for 1 time step.
        """
        error = r - y

        self._potential  = (self._P * error)
        self._integral   = (self._I * self._errorSum)
        self._derivative = (self._D * (error - self._lastError))
        
        output = self._potential + self._integral + self._derivative

        self._errorSum += error
        self._lastError = error
    
        return output

    def paramAdj(self, P, I, D) :
        """Setter for the P, I and D parameters of the class. If any argument is 'None' then the associated existing 
            parameter remains unchanged. Doesn't return anything.
        """
        if type(P) != None :
            self._P = P
        if type(I) != None :
            self._I = I
        if type(D) != None :
            self._D = D

    def getAuxiliaryInfo(self) -> Dict :
        """Returns variables describing this controller as well as the memory of various variables that it has retained.

        Returns
        -------
        Dict
            P          : the P constant the controller was initialized with
            I          : the I constant the controller was initialized with
            D          : the D constant the controller was initialized with
            errorSum   : a running sum of the accumulated error
            lastError  : the most recent error value
            potential  : the most recent potential value (P * lastError)
            integral   : the most recent integral value (I * errorSum)
            derivative : the most recent derivative value (D * the lastError minus second last error)
        """
        return {
            'P'           : self._P,
            'I'           : self._I,
            'D'           : self._D,
            'errorSum'    : self._errorSum,
            'lastError'   : self._lastError,
            'potential'   : self._potential,
            'integral'    : self._integral,
            'derivative'  : self._derivative
        }

class DNN(controllerBase) :
    """A controller entirely comprised of a simple feedforward DNN. Input is a vector of {r[t], r[t-1], r[t-2],..., 
    r[t-n], y[t], y[t-1], y[t-2],..., y[t-n]}. Output is a scalar u[t]. Number of hidden layers and their sizes are left 
    as vairables. Primarily designed to be used in the RTCFL circuit, but general enough to be used elsewhere.

    Parent : controllerBase
    """
    
    def __init__(self, name: str="controller_nn", params: dict=None) :
        """Constructor for DNN plant.

        Parameters
        ----------
        params : dict
            key: "n" -> int : determines the 1D input tensor size, see class description.
            key: "hidden layers" -> list : a list of integers that correspond to size of each hidden layer. must have   
                at least 1 element. 
            key: "activation function" -> torch.nn : Must be a valid pytorch activation function, like torch.nn.ReLU, 
            torch.nn.Sigmoid, torch.nn.tanh, or torch.nn.LeakyReLU.
            key: "learning rate" -> must be a float > 0
            key: "device" -> str : must be either 'cuda' or 'cpu'
        """
        # perform checks to see if arguments are sound
        for arg in ['n', 'hidden layers', 'activation function', 'learning rate', 'device'] :
            if arg not in params :
                raise ValueError(f"controller.DNN class initializer missing an {arg} key in its 'param' dict argument.")
        
        if type(params['n']) != int :
            raise ValueError(f"controller.DNN class initializer requires value for 'n' key in 'param' dict argument to be an int. Right now it is a {type(params['n'])}='{params['n']}'.")

        if type(params['hidden layers']) != list :
            raise ValueError(f"controller.DNN class initializer requires value for 'hidden layers' key in 'param' dict argument to be a list of ints. Right now it is a {type(params['hidden layers'])}='{params['hidden layers']}'.")
        if len(params['hidden layers']) < 1 :
            raise ValueError(f"plant.dnn class initializer requires len for 'hidden layers' key in 'param' dict argument to be at least 1. Right now it is of len='{len(params['hidden layers'])}'.")

        for i, element in enumerate(params['hidden layers']) :
            if type(element) != int :
                raise ValueError(f"controller.DNN class initializer requires value for 'hidden layers' key in 'param' dict argument to be a list of ints. Right now element {i} in said list is a {type(params['hidden layers'][i])}='{params['hidden layers'][i]}'.")

        if type(params['learning rate']) != float :
            raise ValueError(f"plant.dnn class initializer requires value for 'learning rate' key in 'param' dict argument to be of float type. Right now it is a '{type(params['learning rate'])}'.")
        if params['learning rate'] <= 0 :
            raise ValueError(f"plant.dnn class initializer requires value for 'learning rate' key in 'param' dict argument to be greater than 0. Right now it is a '{params['learning rate']}'.")

        valid_devices = ['cuda', 'cpu']
        if type(params['device']) != str or params['device'] not in valid_devices :
            raise ValueError(f"controller.DNN class initializer requires value for 'device' key in 'param' dict argument to be in {valid_devices}. Right now it is a {type(params['device'])}='{params['device']}'.")

        # create internal fields and the pytorch DNN model class (and an instance of it) accordingly
        class NN(torch.nn.Module) :
            def __init__(self) :
                super().__init__()
                # use nn.ModuleList() instead of a simple [] list so that pytorch can index and find the layers.
                self._layerList = torch.nn.ModuleList()
                self._layerList.append(torch.nn.Linear(params['n']*2, params['hidden layers'][0]))
                self._layerList.append(params['activation function']())
                for i in range(0, len(params['hidden layers'])-1) :
                    self._layerList.append(torch.nn.Linear(params['hidden layers'][i], params['hidden layers'][i+1]))
                    self._layerList.append(params['activation function']())
                self._layerList.append(torch.nn.Linear(params['hidden layers'][-1], 1))
                self._layerList.append(params['activation function']())

                self._r_inputMem  = torch.FloatTensor([0.0]*params['n'])
                self._y_inputMem  = torch.FloatTensor([0.0]*params['n'])
                self._inputVector = torch.cat((self._r_inputMem, self._y_inputMem)).to(device=params['device'])

            def forward(self, r, y) :
                """Note that the 'r' and 'y' inputs here must be a scalar. The input memory vector is handled
                internally.
                """
                self._r_inputMem = torch.cat((torch.FloatTensor([r]), self._y_inputMem[:-1]))
                self._y_inputMem = torch.cat((torch.FloatTensor([y]), self._y_inputMem[:-1]))
                self._inputVector = torch.cat((self._r_inputMem.to(device=params['device']), 
                                               self._y_inputMem.to(device=params['device'])))

                output = self._layerList[0](self._inputVector)
                for i in range(1, len(self._layerList)) :
                    output = self._layerList[i](output)
                return output

            def getMemory(self) -> dict :
                """Returns a dict containing the currently maintain input memory vector.
                """
                return {
                    "_r_inputMem"  : self._r_inputMem,
                    "_y_inputMem"  : self._y_inputMem,
                    "_inputVector" : self._inputVector
                }

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

    def execute(self, r, y) -> torch.FloatTensor :
        """Passes a 'r' and 'y' inputs into the controller and returns an output, so straightforward inference. Sets 
        the internal NN model to eval mode. Returns a tensor of the output.

        Parameters
        ----------
        r : int or float
            must be a scalar value. represents 'r[t]'.
        y : int or float
            must be a scalar value. represents 'y[t]'.

        Returns
        -------
        torch.FloatTensor
            output of DNN
        """
        # set the pytorch model module into evaluation mode (disables any dropout layers) and alters normalization
        # characteristics
        self._model.eval()
        return self._model(r, y)

    def train(self,
              r_t       : torch.FloatTensor,
              y_t       : torch.FloatTensor, 
              printInfo : bool = False 
              ) -> torch.FloatTensor :
        """Does one round of forward-backward pass and weight update. Returns the loss. Note that while this uses the
        mean squared error, this loss function is intended to penalize non-perfect damping in the plant output (i.e.
        over or under damped), from which the DNN controller performance can be optimized.

        Parameters
        ----------
        r_t : torch.FloatTensor
            A tensor containing recorded r[t] inputs into the RTCFL circuit.
        y_t : torch.FloatTensor
            A tensor containing recorded y[t] outputs associated with corresponding r[t] values in the r_t tensor. y[t]
            is a function of the controller_nn (See the RTCFL circuit) and the y_t r_t pair is used to calculate the
            loss to train controller_nn via backprop and gradient descent.
        plantOutput : torch.FloatTensor
            a simple scalar which should be pass straight from the output of a plant NN model 
            (see control_feedback_testbench.plants.DNN)
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
        loss = self._lossFunction(y_t.squeeze(), r_t)
        loss.requires_grad = True
        # if print enabled then print.
        if printInfo :
            print(f"\t{self._instanceName} Controller Training loss: {loss}.")
        # backward pass. 
        loss.backward() # compute the gradients
        self._optimizer.step() # update the weights accordingly

        return loss
    
    def getAuxiliaryInfo(self) :
        """Returns a dictionary of all the private class fields.
        """
        modelInputMem = self._model.getMemory()
        
        return {
            'input vector'        : self._inputLength,
            'hidden layers'       : self._hiddenLayersLayout,
            'activation function' : self._activationFunction,
            'learning rate'       : self._learningRate,
            'device'              : self._device,
            'model'               : self._model,
            'model input memory'  : modelInputMem["_inputVector"],
            'model r input mem'   : modelInputMem["_r_inputMem"],
            'model y input mem'   : modelInputMem["_y_inputMem"],
            'loss function'       : self._lossFunction
        }


class RNN(controllerBase) :
    """A RNN based controller that is used to achieve good dampening behavior of a plant it is controlling.

       The number of RNN units to stack one after another is variable. The first RNN unit, i=1, has 1 scalar output: out_1[t],
       and 3 scalar inputs: out_1[t-1], r_1[t], y_1[t]. Hidden states are initialized to 0. Every subsequent RNN unit
       (if stackCount > 1) has 1 scalar output: out_i[t], and 2 scalar inputs: out_i[t-1], out_i-1[t]

       Training is done by pushing a batch of 'r' sequences through the model. A batch can have sequences of any length, 
       but long consistently sized sequences are preferred. For each 'r' value in a sequence, the loss is calculated by
       getting the squared difference between a black box plant and the output of the RNN. This loss of each 'r' value
       is summed, and that is the loss for the given sequence. The losses for all the sequences in the batch are then
       averaged, from which the final gradients are calculated.

       Parent : plantBase
    """

    def __init__(self, name: str = "plant_nn", stackCount: int = 1, learningRate: int=0.01, device: str='cpu') :
        """Constructor for RNN controller.

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
        if stackCount < 1 :
            raise ValueError(f"controller.RNN class initializer needs 'stackCount' argument to be >= 1. Now it is '{stackCount}'.")
        if learningRate <= 0 :
            raise ValueError(f"controller.RNN class initializer needs 'learningRate' argument to be > 0. Now it is '{learningRate}'.")

        # create RNN model class.
        class RNN(torch.nn.Module) :
            def __init__(self):
                super().__init__()

                self._FirstLinearLayer = torch.nn.Linear(3, 1)
                self._LinearLayer      = torch.nn.Linear(2, 1)
                self._ReLULayer        = torch.nn.ReLU()
                self._hiddenStateList  = []
                for i in range(stackCount) :
                    self._hiddenStateList.append(torch.FloatTensor([0]))

            def forward(self, r: float, y: float) :
                """
                Parameters
                ----------
                r : int
                    'r' must be scalar
                """
                output = self._ReLULayer(self._FirstLinearLayer(torch.cat((torch.FloatTensor([r]).to(device=device), 
                                                                           torch.FloatTensor([y]).to(device=device),
                                                                           self._hiddenStateList[0].to(device=device)
                                        ))))
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

    def execute(self, r: float, y: float) -> torch.FloatTensor :
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
        return self._model(r, y)

    def train(self, r_sequences: list, plant_nn_function: plantBase.execute) -> torch.FloatTensor :
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
            The function pointer used to calculate the plant neural net outputs from which to calculate the loss
            function. Note that this should return a tensor with requires_grad = True so that gradients can be traced.

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
        loss_sum = torch.tensor(0, dtype=float)
        loss_sum.requires_grad = True
        for sequence in r_sequences :
            lossList = torch.tensor([])
            for r in sequence :
                newloss = (r - plant_nn_function(r))**2
                lossList = torch.cat((lossList, newloss))
            loss_sum = torch.sum(lossList)
        loss = loss_sum / len(r_sequences)
        # backward pass
        loss.backward(retain_graph=True)  # compute the gradients
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