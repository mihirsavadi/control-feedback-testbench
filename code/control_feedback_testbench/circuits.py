from multiprocessing.sharedctypes import Value
from lib import *

from control_feedback_testbench import base, plants, controllers

## BASE CIRCUIT CLASS HERE #############################################################################################
class circuitBase(abc.ABC) :
    """Base class for circuits. Children of this class will provide a framework to implement specific interconnections
    of specific controllers, plants, and other actors. The outputs of circuit classes need to be standardized and
    homogenous for the sake of modularity in the scheduler class which will compose an instance of this circuit class --
    the abstract methods and fields defines in this abstract base class define these standardizations. However the actor
    classes that circuit classes will compose/employ can exploit any of the class-specific I/O's to build the intended
    specifics of the circuit -- any I/O standardization/homogenizations from the plant classes are purely intended make
    building of circuit classes simpler and more predictable.
    """

    @abc.abstractmethod
    def __init__(self) :
        """Constructor. In child classes must instantiate the actors necessary to build the child's circuit here. If 
        applicable add additional lists to self.returnDict which is instantiated here, e.g. values generated in the
        circuit that the designer wishes to observe that arent e, u, or y. Additional arguments may be given for -
        either static or starting values for dynamic - parameters required of a circuit's actors. In order to reduce
        confusion between the chain of child classes, for circuits never use default arguments for parameters in
        constructors.
        """

        # This dict is important. It is public. The default values are e,u, and y. Depending on the child circuit
        # implementation, there can be more. The user should call this field in order to retrieve all information to see
        # what is going on the circuit since the self.execute().
        self.returnDict = {
            'r' : [],
            'e' : [],
            'u' : [],
            'y' : []
        }

        # must have the following list for the scheduler super class to know which plots to plot
        self.keysToPlot = ['r', 'u', 'y', 'e']

    @abc.abstractmethod
    def __str__(self):
        """String return overide for this class. Must be whatever the verbose name of this circuit is.
        """
        return 'circuit base class.'

    @abc.abstractmethod
    def getDescriptionString(self) -> str:
        """Returns a string summarizing the parameters of each actor and other pertinent details of the particular 
           circuit.
        """
        return 'baseclass, no description.'

    @abc.abstractmethod
    def execute(self, r: float) -> float:
        """In child classes their circuit for one time step should be implemented here -- actors are executed with the
        first one being fed r, and subsequent actors are fed the outputs of previous actors. The lists in
        self.returnDict are filled in accordingly. Actors are responsible for handling their own memory of input values.
        In this parent class this remains empty.

        Parameters
        ----------
        r: float
            the r_input required to be fed into the circuit

        Returns
        -------
        Float
            the final output 'y' of the plant
        """
        y = -6969
        return y

    def generatePlot(self, printPlot, destination, plotXLimit: tuple=None) -> plt :
        """Generates a plot from whatever is in the returnDict dictionary. OPTIONAL: can overide the default 
        scheduler behavior for generating a plot by calling this function, if desired.

        Parameters
        ----------
        printPlot : bool, optional
            If True then print a jpg of the plot with the destination specificed by the argument 'destination', by
            default False
        destination : str, optional
            The location to print the plot. Needs to include the output file and format. Stick to .jpg or .png to be
            safe, by default './plot.jpg'
        plotXLimit : tuple, optional
            By default this is None. Otherwise must be a tuple where the first element is the lower limit for the x axis
            to be plotted and the second element is the upper limit for the x axis to be plotted. Lower limit must never
            be less than 0, and upper limit must never be more than the maximum length of the plot -- errors can occur
            if user is not careful.

        Returns
        -------
        plt
            returns the matplotlib plt object generated.
        """
        return None


########################################################################################################################
######################################### CHILD CIRCUIT CLASSES BELOW ##################################################
########################################################################################################################

########################################################################################################################
## STATIC PID-FOPDT CLASS BELOW
########################################################################################################################
class PID_static_FOPDT(circuitBase):
    """This class implements the a simple static PID controller that controls a simple FOPDT. Make sure to see comments
    for each function in the parent class for context on what they are supposed to do.

    Parent : circuitBase
    """
    
    def __init__(self, P: float, I: float, D: float, K: float, tau: float, theta: float) :
        """In addition to parent, requires the static parameters for the PID controller and FOPDT plant used in this
        circuit.

        Parameters
        ----------
        P : float
            potential (P) parameter for the PID controller.
        I : float
            integral (I) parameter for the PID controller.
        D : float
            derivative (D) parameter for the PID controller.
        K : float
            process gain (K_p) parameter for the FOPDT plant model.
        tau : float
            process time constant (tau_p) parameter for the FOPDT plant model.
        theta : float
            process dead time (theta_p) parameter for the FOPDT plant model.
        """
        super().__init__()

        print(f"PID_static_FOPDT circuit created! Parameters are:\n\tP\t: {P}\n\tI\t: {I}\n\tD\t: {D}\n\tK\t: {K}\n\ttau\t: {tau}\n\ttheta\t: {theta}")

        # establish fields here
        self.returnDict["P"]           = P
        self.returnDict["I"]           = I
        self.returnDict["D"]           = D
        self.returnDict["K"]           = K
        self.returnDict["tau"]         = tau
        self.returnDict["theta"]       = theta

        self.keysToPlot = ['r', 'u', 'y', 'e']
        
        # intiate actors here
        self._controller = controllers.PID_static(
            params={
                "P" : self.returnDict["P"],
                "I" : self.returnDict["I"],
                "D" : self.returnDict["D"]
            }
        )

        self._plant = plants.FOPDT(
            params={
                "K"     : self.returnDict["K"],
                "tau"   : self.returnDict["tau"],
                "theta" : self.returnDict["theta"]
            }
        )

    def __str__(self):
        """String return overide for this class. Just a quick descriptive name for this circuit..
        """
        return 'Static PID to FOPDT Circuit.'

    def getDescriptionString(self) -> str:
        """Text summary of controller PID actor and plant FOPDT actor parameters.
        """
        return f"PID params: P={self.returnDict['P']}, I={self.returnDict['I']}, D={self.returnDict['D']}.\nFOPDT params: K={self.returnDict['K']}, tau={self.returnDict['tau']}, theta={self.returnDict['theta']}."


    def execute(self, r) -> float:
        """Executes the PID_static_FOPDT circuit and updates lists contained in self.returnDict. Note that all the lists
        in this dict need to be of the same length, otherwise plot generation in the scheduler class will break. This
        implies contextual temporal relevance between corresponding indexes of each element in these lists.

        Parameters
        ----------
        r: float
            the r_input required to be fed into the circuit

        Returns
        -------
        Float
            the final output 'y' of the plant
        """

        last_yVal = 0
        if len(self.returnDict['y']) > 0 :
            last_yVal = self.returnDict['y'][-1]
        output_controller = self._controller.execute(r, last_yVal)
        output_plant = self._plant.execute(output_controller)

        self.returnDict['r'].append(r)
        self.returnDict['e'].append(self._controller.getAuxiliaryInfo()['lastError'])
        self.returnDict['u'].append(output_controller)
        self.returnDict['y'].append(output_plant)

        return output_plant
    

########################################################################################################################
## RTCFL_DNN Circuit CLASS BELOW
########################################################################################################################

class RTCFL_DNN(circuitBase) :
    """This class implements the RTCFL circuit using DNN's for the NN plants. Tested to perform like garbage, or tuning 
    this one itself such a pain in the ass that it defeats the goal of the RTCFL in the first place -- avoid tuning 
    PID's.

    Parent : circuitBase
    """

    def __init__(self, 
                 # plant params here
                 plant_bb       : base.plantBase,
                 plant_nn       : base.plantBase,
                 plant_minError : float,
                 plant_k        : int,
                 plant_l        : int,
                 # controller params here
                 controller_nn       : base.controllerBase,
                 controller_minLoss : float,
                 controller_k        : int,
                 controller_l        : int
                 ):
        """Class initializer. Requires a fair amount of arguments - there are no defaults.

        Parameters
        ----------
        plant_bb : control_feedback_testbench.base.plantBase
            The black box plant that the plantNN will have to emulate with low error. This object simply takes in a 
            scalar input and spites out a scalar output. If desired, an object can be placed here that provides a data
            pipeline to an actual real world model, instead of a simulated plant object from the 
            control_feedback_testbench.plants module which would be used in a typical case.
        plant_nn : control_feedback_testbench.base.plantBase
            The NN to represent the plant_nn. Must be a plant actor with a train() method.
        plant_minloss : float
            The minimum mean squared error loss value after which the plantNN will be deemed accurate enough and the 
            RTCFL can move on to training the controllerNN. Must be > 0. 
        plant_k : int
            At every k | k ≥ 1 time-steps, an internal data base is appended to with y[t] and the associated label 
            (that is plant_bb's output).
        plant_l : int
            At every l | l = kp, p ≥ 1, p ∈ Z time-steps: the average loss_plnt is calculated from the data base, which
            plant_NN then uses to undergo a round of gradient descent and update its parameters; and the database is
            reset to be empty. This is repeated until loss_plnt reaches a minimum threshold value, plant_minloss.
        controller_nn : control_feedback_testbench.base.plantBase
            The NN to represent the controller_nn. Must be a plant actor with a train() method.
        controller_minloss : float
            The minimum mean squared error loss value after which the controllerNN will be deemed accurate enough and 
            the RTCFL will have reached optimal control convergence. Must be > 0.
        controller_k : int
            At every k | k ≥ 1 time-steps, an internal database is appended to with associated y[t] and r[t] as 
            data-points.
        controller_l : int
            At every l | l = kp, p ≥ 1, p ∈ Z time-steps: the average loss_ctrlr is calculated from the data base, which
            controller_NN then uses to undergo a round of gradient descent and update its parameters; and the database
            is reset to be empty. This is repeated until loss_ctrlr reaches a minimum threshold value,
            controller_minloss, after which our RTCFL will have reached optimal performance!
        """
        super().__init__()

        # perform checks to see if arguments are sound. Dont need to check for variable types, only value ranges.
        if plant_minError <= 0 :
            raise ValueError(f"circuits.RTCFL class initializer requires argument 'plant_minloss' to be >0. Right now it is ={plant_minError}")

        if plant_k < 1 :
            raise ValueError(f"circuits.RTCFL class initializer requires argument 'plant_k' to be >=1. Right now it is '{plant_k}'.")
        if plant_l < plant_k or (plant_l < plant_k and plant_k % plant_l != 0):
            raise ValueError(f"circuits.RTCFL class initializer requires argument 'plant_l' to obey 'l | l = kp, p ≥ 1, p ∈ Z'. Right now it is '{plant_l}'.")

        if plant_minError <= 0 :
            raise ValueError(f"circuits.RTCFL class initializer requires argument 'controller_minloss' to be >0. Right now it is ={plant_minError}")

        if controller_k < 1 :
            raise ValueError(f"circuits.RTCFL class initializer requires argument 'controller_k' to be >=1. Right now it is '{controller_k}'.")
        if controller_l < controller_k or (controller_l < controller_k and controller_k % controller_l != 0) :
            raise ValueError(f"circuits.RTCFL class initializer requires argument 'controller_l' to obey 'l | l = kp, p ≥ 1, p ∈ Z'. Right now it is '{controller_l}'.")
        
        # instantiate private field objects for actors here.
        self.plant_bb = plant_bb
        self.plant_nn = plant_nn
        self.controller_nn = controller_nn

        # instantiate public return dict field here to keep track of memory for each part of circuit, and other details.
        # remember that 'r', 'e', 'u', 'y' keys already instantiated in circuit base class, but 'e' not relevant here.
        # make sure all these lists are always the same length. If 'label' for example is no longer being generated
        # because the rtcfl algorithm has moved on from plant_nn training, just keeping appending the last known value;
        # or if 'loss_ctrlr' has not been appended to it yet, just append 'None' or some large number. 
        self.returnDict['loss_plnt']  = [] # make sure to enter floats in here not tensors
        self.returnDict['loss_ctrlr'] = [] # make sure to enter floats in here not tensors
        self.returnDict['label']      = []
        self.returnDict['mode']       = []

        # instantiate private scalar fields to keep track of things. The weird flag numbers are just so that they can be
        # visible in a plot.
        self._circuitModeFlag = -100   # -100: step1 train plant_nn; 0: step2 train controller_nn; 100: RTCFL optimal now.

        self._plant_DB_label  = []
        self._plant_DB_y      = []
        self._controller_DB_y = []
        self._controller_DB_r = []

        self._totalTimeStepCount = 0

        # instantiate other private fields here.
        self._plant_minError     = plant_minError
        self._plant_k            = plant_k
        self._plant_l            = plant_l

        self._controller_minLoss = controller_minLoss
        self._controller_k       = controller_k
        self._controller_l       = controller_l

        self.keysToPlot = ['r', 'u', 'y', 'e', 'label', 'loss_plnt', 'loss_ctrlr', 'mode']
        
        # Print CLI message on circuit successful instantiation here.
        print(f"DNN based RTCFL Circuit instantiated! {self.getDescriptionString()[10:]}")

    def __str__(self):
        """String return overide for this class. Just a quick descriptive name for this circuit..
        """
        return 'DNN based RTCFL Circuit.'

    def getDescriptionString(self) -> str :
        """Text summary of circuit parameters.
        """
        return f"DNN based RTCFL Parameters are:\n\tPlant_bb : {self.plant_bb}\n\tPlant_NN model : {self.plant_nn.getAuxiliaryInfo()['model']}\n\tPlant MinLoss : {self._plant_minError}\n\tPlant k : {self._plant_k }\n\tPlant l : {self._plant_l }\n\tPlant_bb model : {self.plant_bb}\n\tController_NN model : {self.controller_nn.getAuxiliaryInfo()['model']}\n\tController MinLoss : {self._controller_minLoss}\n\tController k : {self._controller_k}\n\tController l : {self._controller_l }"

    def execute(self, r: float) -> float :
        """Takes in a new r value and updates the RTCFL circuits time-step by time-step.

        Parameters
        ----------
        r: float
            the r_input required to be fed into the circuit.

        Returns
        -------
        Float
            the final output 'y' of the plant.
        """
        # first do some counting related house keeping
        self._totalTimeStepCount += 1

        # get last value of y first
        last_yVal = 0
        if len(self.returnDict['y']) > 0 :
            last_yVal = self.returnDict['y'][-1]

        # Check mode flag. Step 1: train the plant_nn
        if self._circuitModeFlag == -100 :
            # first store current mode
            self.returnDict['mode'].append(self._circuitModeFlag) 

            # then get actor outputs.
            output_plant_bb = self.plant_bb.execute(r)
            output_controller = self.controller_nn.execute(r, last_yVal)
            output_plant_nn = self.plant_nn.execute(r)
            
            # Append to DB's at every k
            if (self._totalTimeStepCount % self._plant_k) == 0 :
                self._plant_DB_label.append(output_plant_bb)
                self._plant_DB_y.append(output_plant_nn)

            # At every l do a round of training on plant_nn using DB's and reset DB's. Also check if loss is good to
            # progress mode.
            updateLossReturnDictLists = True
            if (self._totalTimeStepCount % self._plant_l) == 0 :
                loss = self.plant_nn.train(torch.FloatTensor(self._plant_DB_y), 
                                           torch.FloatTensor(self._plant_DB_label), 
                                           printInfo=False)
                self._plant_DB_label  = []
                self._plant_DB_y      = []

                # if loss < self._plant_minError :
                if abs(r - output_plant_nn.item()) < self._plant_minError :
                    self._circuitModeFlag = 0

                updateLossReturnDictLists = False
                self.returnDict['loss_plnt'].append(loss)
                if len(self.returnDict['loss_ctrlr']) == 0 :
                    self.returnDict['loss_ctrlr'].append(0.0)
                else :
                    self.returnDict['loss_ctrlr'].append(self.returnDict['loss_ctrlr'][-1])

            # Append to the remaining returnDict lists and return
            if updateLossReturnDictLists :
                if len(self.returnDict['loss_plnt']) == 0 :
                    self.returnDict['loss_plnt'].append(0.0)
                    self.returnDict['loss_ctrlr'].append(0.0)
                else :
                    self.returnDict['loss_plnt'].append(self.returnDict['loss_plnt'][-1])
                    self.returnDict['loss_ctrlr'].append(self.returnDict['loss_ctrlr'][-1])
            self.returnDict['label'].append(output_plant_bb)
            self.returnDict['r'].append(r)
            self.returnDict['u'].append(output_controller)
            self.returnDict['y'].append(output_plant_nn.item())
            self.returnDict['e'].append(self.returnDict['r'][-1] - self.returnDict['y'][-1])
            return output_plant_nn.item()

        # Check mode flag. Step 2: train the controller_nn if Step 1 is done
        elif self._circuitModeFlag == 0 :
            # first store current mode
            self.returnDict['mode'].append(self._circuitModeFlag) 

            # then get actor outputs.
            output_plant_bb = self.plant_bb.execute(r)
            output_controller = self.controller_nn.execute(r, last_yVal)
            output_plant_nn = self.plant_nn.execute(output_controller)

            # Append to DB's at every k
            if (self._totalTimeStepCount % self._controller_k) == 0 :
                self._controller_DB_y.append(output_plant_nn)
                self._controller_DB_r.append(r)

            # At every l do a round of training on plant_nn using DB's and reset DB's. Also check if loss is good to
            # progress mode.
            updateLossReturnDictLists = True
            if (self._totalTimeStepCount % self._controller_l) == 0 :
                loss = self.controller_nn.train(torch.FloatTensor(self._controller_DB_r), 
                                                torch.FloatTensor(self._controller_DB_y), 
                                                printInfo=False)
                self._controller_DB_y = []
                self._controller_DB_r = []

                if loss < self._controller_minLoss :
                    self._circuitModeFlag = 100

                updateLossReturnDictLists = False
                self.returnDict['loss_ctrlr'].append(loss)
                if len(self.returnDict['loss_plnt']) == 0 :
                    self.returnDict['loss_plnt'].append(0.0)
                else :
                    self.returnDict['loss_plnt'].append(self.returnDict['loss_plnt'][-1])

            # Append to the remaining returnDict lists and return
            if updateLossReturnDictLists :
                if len(self.returnDict['loss_ctrlr']) == 0 :
                    self.returnDict['loss_plnt'].append(0.0)
                    self.returnDict['loss_ctrlr'].append(0.0)
                else :
                    self.returnDict['loss_plnt'].append(self.returnDict['loss_plnt'][-1])
                    self.returnDict['loss_ctrlr'].append(self.returnDict['loss_ctrlr'][-1])
            self.returnDict['label'].append(output_plant_bb)
            self.returnDict['r'].append(r)
            self.returnDict['u'].append(output_controller)
            self.returnDict['y'].append(output_plant_nn.item())
            self.returnDict['e'].append(self.returnDict['r'][-1] - self.returnDict['y'][-1])
            return output_plant_nn.item()

        # Check mode flag. Otherwise operate as NN's already trained.
        elif self._circuitModeFlag == 100 :
            # first store current mode
            self.returnDict['mode'].append(self._circuitModeFlag) 

            # then find and store actor outputs
            output_plant_bb = self.plant_bb.execute(r)
            output_controller = self.controller_nn.execute(r, last_yVal)
            output_plant_nn = self.plant_nn.execute(output_controller)

            # Append to the remaining returnDict lists and return
            self.returnDict['loss_plnt'].append(self.returnDict['loss_plnt'][-1])
            self.returnDict['loss_ctrlr'].append(self.returnDict['loss_ctrlr'][-1])
            self.returnDict['label'].append(output_plant_bb)
            self.returnDict['r'].append(r)
            self.returnDict['u'].append(output_controller)
            self.returnDict['y'].append(output_plant_nn.item())
            self.returnDict['e'].append(self.returnDict['r'][-1] - self.returnDict['y'][-1])
            return output_plant_nn.item()

        # a bit of bug prevention here
        else :
            raise ValueError(f"'self._circuitModeFlag should only be -100, 0, or 100. Now it is '{self._circuitModeFlag}'. Some bug type shit is going on here fam, hop in and fix it.")


########################################################################################################################
## RTCFL_RNN Circuit CLASS BELOW
########################################################################################################################

class RTCFL_RNN(circuitBase) :
    """This class implements the RTCFL circuit using RNN's for the NN plants.

    Parent: circuitBase
    """

    def __init__(self, 
                 plant_bb_model                  : plants.plantBase,
                 plant_nn_stackCount             : int=1,
                 plant_nn_learningRate           : float=0.01,
                 plant_nn_trainSeqLen            : int=50,
                 plant_nn_trainSeqAddPeriod      : int=50,
                 plant_nn_trainSeqDBSize         : int=100,
                 plant_nn_minLoss                : float=3,
                 controller_nn_stackCount        : int=1,
                 controller_nn_learningRate      : float=0.01,
                 controller_nn_trainSeqLen       : int=50,
                 controller_nn_trainSeqAddPeriod : int=50,
                 controller_nn_trainSeqDBSize    : int=100,
                 controller_nn_minLoss           : float=3,
                 device: str='cpu'
                ):
        """Class initializer.

        Parameters
        ----------
        plant_bb_model : plants.plantBase
            Black box plant model object for the plant_nn to converge behavior to emulate. The black box plant that the 
            plantNN will have to emulate with low error. This object simply takes in a scalar input and spites out a 
            scalar output. If desired, an object can be placed here that provides a data pipeline to an actual real 
            world model, instead of a simulated plant object from the control_feedback_testbench.plants module which 
            would be used in a typical case.
        plant_nn_stackCount : int, optional
            The number of RNN unit stacks in the plant_nn network, by default 1
        plant_nn_learningRate : float, optional
            The learning rate of the plant_nn, by default 0.01
        plant_nn_trainSeqLen : int, optional
            The length of each training sequence to be collected into the training database as inputs come in. At every
            'plant_nn_trainSeqAddPeriod' the sequence {r[t], r[t-1],...,r[t-plant_nn_trainSeqLen]} will be added to the
            training database., by default 50. 
        plant_nn_trainSeqAddPeriod : int, optional
            The period at which training sequences are added into the training database. At every
            'plant_nn_trainSeqAddPeriod' the sequence {r[t], r[t-1],...,r[t-plant_nn_trainSeqLen]} will be added to the
            training database., by default 50. Must be >= plant_nn_trainSeqLen
        plant_nn_trainSeqDBSize : int, optional
            The number of training sequence to be added to the database before one round of training and weight updates
            occurs. When this is reached, the training database is reset to empty. , by default 100
        plant_nn_minLoss : float, optional
            The minimum loss after which the RTCFL circuit will stop training the plant_NN and will move on to training
            the controller_nn
        controller_nn_stackCount : int, optional
            The number of RNN unit stacks in the controller_nn network, by default 1
        controller_nn_learningRate : float, optional
            The learning rate of the ontroller_nn, by default 0.01
        controller_nn_trainSeqLen : int, optional
            The length of each training sequence to be collected into the training database as inputs come in. At every
            'controller_nn_trainSeqAddPeriod' the sequence {r[t], r[t-1],...,r[t-controller_nn_trainSeqLen]} will be 
            added to the training database., by default 50.
        controller_nn_trainSeqAddPeriod : int, optional
            The period at which training sequences are added into the training database. At every
            'controller_nn_trainSeqAddPeriod' the sequence {r[t], r[t-1],...,r[t-controller_nn_trainSeqLen]} will be
            added to the training database., by default 50. Must be >= controller_nn_trainSeqLen
        controller_nn_minLoss: float, optional
            The minimum loss after which the RTCFL circuit will stop training the controller_NN and the RTCFL circuit
            will be optimized and operating in pure inference mode.
        device : str, optional
            Can be either 'cpu' or 'cuda', by default 'cpu'
        """
        super().__init__()

        # do some hazard prevention here
        if plant_nn_trainSeqAddPeriod < plant_nn_trainSeqLen :
            raise ValueError(f"Argument 'plant_nn_trainSeqAddPeriod' (currently={plant_nn_trainSeqAddPeriod}) must be >= than 'plant_nn_trainSeqLen' (currently={plant_nn_trainSeqLen})")
        if controller_nn_trainSeqAddPeriod < controller_nn_trainSeqLen :
            raise ValueError(f"Argument 'plant_nn_trainSeqAddPeriod' (currently={controller_nn_trainSeqAddPeriod}) must be >= than 'plant_nn_trainSeqLen' (currently={controller_nn_trainSeqLen})")

        # instantiate private field objects for actors here.
        self.plant_bb = plant_bb_model
        self.plant_nn = plants.RNN(name="plant_nn", 
                                   stackCount=plant_nn_stackCount,
                                   learningRate=plant_nn_learningRate,
                                   device=device)
        self.controller_nn = controllers.RNN(name="plant_nn", 
                                             stackCount=controller_nn_stackCount,
                                             learningRate=controller_nn_learningRate,
                                             device=device)

        # instantiate public return dict field here to keep track of memory for each part of circuit, and other details.
        # remember that 'r', 'e', 'u', 'y' keys already instantiated in circuit base class, but 'e' not relevant here.
        # make sure all these lists are always the same length. If 'label' for example is no longer being generated
        # because the rtcfl algorithm has moved on from plant_nn training, just keeping appending the last known value;
        # or if 'loss_ctrlr' has not been appended to it yet, just append 'None' or some large number. 
        self.returnDict['loss_plnt']  = [] # make sure to enter floats in here not tensors
        self.returnDict['loss_ctrlr'] = [] # make sure to enter floats in here not tensors
        self.returnDict['label']      = []
        self.returnDict['mode']       = []

        # instantiate other private fields here.
        self._plant_nn_trainSeqLen            = plant_nn_trainSeqLen
        self._plant_nn_trainSeqAddPeriod      = plant_nn_trainSeqAddPeriod
        self._plant_nn_trainSeqDBSize         = plant_nn_trainSeqDBSize
        self._plant_nn_minLoss                = plant_nn_minLoss

        self._controller_nn_trainSeqLen       = controller_nn_trainSeqLen
        self._controller_nn_trainSeqAddPeriod = controller_nn_trainSeqAddPeriod
        self._controller_nn_trainSeqDBSize    = controller_nn_trainSeqDBSize
        self._controller_nn_minLoss           = controller_nn_minLoss

        # instantiate private scalar fields to keep track of things. The weird flag numbers are just so that they can be
        self._circuitModeFlag = -100 # -100: step1 train plant_nn; 0: step2 train controller_nn; 100: RTCFL optimal now.

        self._plant_sequencesDB      = []
        self._controller_sequencesDB = []

        self._totalTimeStepCount = 0

        self.keysToPlot = ['r', 'u', 'y', 'e', 'label', 'loss_plnt', 'loss_ctrlr', 'mode']

        # Print CLI message on circuit successful instantiation here.
        print(f"RNN based RTCFL Circuit instantiated! {self.getDescriptionString()[6:]}")

    def __str__(self):
        """String return overide for this class. Just a quick descriptive name for this circuit..
        """
        return 'RNN based RTCFL Circuit.'

    def getDescriptionString(self) -> str :
        """Text summary of circuit parameters.
        """
        return f"RNN based RTCFL Parameters are:\n\tplant_bb_model : {self.plant_bb}\n\tplant_NN_model : {self.plant_nn.getAuxiliaryInfo()['model']}\n\tcontroller_NN_model : {self.controller_nn.getAuxiliaryInfo()['model']}\n\tplant_nn_trainSeqLen : {self._plant_nn_trainSeqLen}\n\tplant_nn_trainSeqAddPeriod : {self._plant_nn_trainSeqAddPeriod}\n\tplant_nn_trainSeqDBSize : {self._plant_nn_trainSeqDBSize}\n\tplant_nn_minLoss : {self._plant_nn_minLoss}\n\tcontroller_nn_trainSeqLen : {self._controller_nn_trainSeqLen}\n\tcontroller_nn_trainSeqAddPeriod : {self._controller_nn_trainSeqAddPeriod}\n\tcontroller_nn_trainSeqDBSize : {self._controller_nn_trainSeqDBSize}\n\tcontroller_nn_minLoss : {self._controller_nn_minLoss}"

    def execute(self, r: float) -> float :
        """Takes in a new r value and updates the RTCFL circuits time-step by time-step.

        Parameters
        ----------
        r: float
            the r_input required to be fed into the circuit.

        Returns
        -------
        Float
            the final output 'y' of the plant.
        """
        # first do some counting related house keeping
        self._totalTimeStepCount += 1

        # get last value of y first
        last_yVal = 0
        if len(self.returnDict['y']) > 0 :
            last_yVal = self.returnDict['y'][-1]

        # Check mode flag. Step 1: train the plant_nn
        if self._circuitModeFlag == -100 :
            # first store current mode
            self.returnDict['mode'].append(self._circuitModeFlag)

            # then get actor outputs.
            output_plant_bb   = self.plant_bb.execute(r)
            output_controller = self.controller_nn.execute(r, last_yVal)
            output_plant_nn   = self.plant_nn.execute(r)

            # append to DB at every _trainSeqAddPeriod
            if (self._totalTimeStepCount % self._plant_nn_trainSeqAddPeriod) == 0 :
                self._plant_sequencesDB.append(self.returnDict['r'][-self._plant_nn_trainSeqLen:])

            # if DB size reached, commence training, get loss value, then reset database.
            updateLossReturnDictLists = True
            if len(self._plant_sequencesDB) == self._plant_nn_trainSeqDBSize :
                loss = self.plant_nn.train(self._plant_sequencesDB, self.plant_bb.execute)
                self._plant_sequencesDB = []

                # update mode if loss is less than plant_nn_minLoss 
                if loss < self._plant_nn_minLoss :
                    self._circuitModeFlag = 0

                updateLossReturnDictLists = False
                self.returnDict['loss_plnt'].append(loss)
                if len(self.returnDict['loss_ctrlr']) == 0 :
                    self.returnDict['loss_ctrlr'].append(-1)
                else :
                    self.returnDict['loss_ctrlr'].append(self.returnDict['loss_ctrlr'][-1])

            # Append to the remaining returnDict lists and return
            if updateLossReturnDictLists :
                if len(self.returnDict['loss_plnt']) == 0 :
                    self.returnDict['loss_plnt'].append(-1)
                    self.returnDict['loss_ctrlr'].append(-1)
                else :
                    self.returnDict['loss_plnt'].append(self.returnDict['loss_plnt'][-1])
                    self.returnDict['loss_ctrlr'].append(self.returnDict['loss_ctrlr'][-1])
            self.returnDict['label'].append(output_plant_bb)
            self.returnDict['r'].append(r)
            self.returnDict['u'].append(output_controller)
            self.returnDict['y'].append(output_plant_nn.item())
            self.returnDict['e'].append(self.returnDict['r'][-1] - self.returnDict['y'][-1])

            return output_plant_nn.item()


        # Check mode flag. Step 2: train the controller_nn if Step 1 is done
        elif self._circuitModeFlag == 0 :
            # first store current mode
            self.returnDict['mode'].append(self._circuitModeFlag)

            # then get actor outputs.
            output_plant_bb = self.plant_bb.execute(r)
            output_controller = self.controller_nn.execute(r, last_yVal)
            output_plant_nn = self.plant_nn.execute(output_controller)

            # append to DB at every _trainSeqAddPeriod
            if (self._totalTimeStepCount % self._controller_nn_trainSeqAddPeriod) == 0 :
                self._controller_sequencesDB.append(self.returnDict['r'][-self._controller_nn_trainSeqLen:])

            # if DB size reached, commence training, get loss value, then reset database.
            updateLossReturnDictLists = True
            if len(self._controller_sequencesDB) == self._controller_nn_trainSeqDBSize :
                loss = self.controller_nn.train(self._controller_sequencesDB, self.plant_nn.execute)
                self._controller_sequencesDB = []

                # update mode if loss is less than controller_nn_minLoss 
                if loss < self._controller_nn_minLoss :
                    self._circuitModeFlag = 100

                updateLossReturnDictLists = False
                self.returnDict['loss_ctrlr'].append(loss)
                if len(self.returnDict['loss_plnt']) == 0 :
                    self.returnDict['loss_plnt'].append(-1)
                else :
                    self.returnDict['loss_plnt'].append(self.returnDict['loss_plnt'][-1])

            # Append to the remaining returnDict lists and return
            if updateLossReturnDictLists :
                if len(self.returnDict['loss_ctrlr']) == 0 :
                    self.returnDict['loss_plnt'].append(-1)
                    self.returnDict['loss_ctrlr'].append(-1)
                else :
                    self.returnDict['loss_plnt'].append(self.returnDict['loss_plnt'][-1])
                    self.returnDict['loss_ctrlr'].append(self.returnDict['loss_ctrlr'][-1])
            self.returnDict['label'].append(output_plant_bb)
            self.returnDict['r'].append(r)
            self.returnDict['u'].append(output_controller)
            self.returnDict['y'].append(output_plant_nn.item())
            self.returnDict['e'].append(self.returnDict['r'][-1] - self.returnDict['y'][-1])

            return output_plant_nn.item()

        # Check mode flag. Otherwise operate as NN's already trained.
        elif self._circuitModeFlag == 100 :
            # first store current mode
            self.returnDict['mode'].append(self._circuitModeFlag) 

            # then find and store actor outputs
            output_plant_bb = self.plant_bb.execute(r)
            output_controller = self.controller_nn.execute(r, last_yVal)
            output_plant_nn = self.plant_nn.execute(output_controller)

            # Append to the remaining returnDict lists and return
            self.returnDict['loss_plnt'].append(self.returnDict['loss_plnt'][-1])
            self.returnDict['loss_ctrlr'].append(self.returnDict['loss_ctrlr'][-1])
            self.returnDict['label'].append(output_plant_bb)
            self.returnDict['r'].append(r)
            self.returnDict['u'].append(output_controller)
            self.returnDict['y'].append(output_plant_nn.item())
            self.returnDict['e'].append(self.returnDict['r'][-1] - self.returnDict['y'][-1])

            return output_plant_nn.item()

        # a bit of bug prevention here
        else :
            raise ValueError(f"'self._circuitModeFlag should only be -100, 1, or 100. Now it is '{self._circuitModeFlag}'. Some bug type shit is going on here fam, hop in and fix it.")