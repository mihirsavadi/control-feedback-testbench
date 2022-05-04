# Mihir Savadi 4 April 2022

from lib import *

class actorBase(abc.ABC) :
    """Base class upon which actors are built.
    """
    
    @abc.abstractmethod
    def __init__(self, params={}):
        """Constructor

        Parameters
        ----------
        staticParams : dict
            a dictionary of static parameters for the actor in question, by default empty
        startingParams : dict, optional
            a dictionary containing the starting values for dynamic parameters for the actor in question, by default
            empty
        """
        pass

    @abc.abstractmethod
    def execute(self, inputs: Dict[str, float]) -> float :
        """This function encapsulates all the calculations an actor needs to do in a single timestep. To do so it will
        call on present and historical inputs if necessary, and will return a single output float. It will also store
        values into whatever fields this actor has that are necessary.

        Parameters
        ----------
        inputs: Dict
            Dict of inputs. Each element in the dictionary should be a single Float. Should by default just have r for
            controllers and e for plants. Depending on the actor's needs, the dict may have to contain more items --
            this needs to be exception enforced.

        Returns
        -------
        Float
            output value for this actor
        """
        output = -69
        return output

    @abc.abstractmethod
    def paramAdj(self, inputs) :
        """Allows for parameter adjustment of specifically allowed parameters/class fields after class has been
           instantiated. 

        Parameters
        ----------
        inputs :
            Inputs can be in any format and should conform to the specifics of the actor.
        """
        pass

    @abc.abstractclassmethod
    def getAuxiliaryInfo(self) -> Dict : 
        """Returns a Dict of all parameters and other information in the actor that isn't an input or output.

        Returns
        -------
        Dict
            Dict for all 'internal' information of the actor.
        """
        outputDict = {
            
        }
        
        return outputDict


class plantBase(actorBase) :
    """PLANT base class, child of actorBase class.

    Parent : actorBase
    """

    @abc.abstractmethod
    def __init__(self, params={}):
        """Initiate plant specific fields here

        Parameters
        ----------
        staticParams : dict, optional
            a dictionary of static parameters for the plant in question, by default empty
        startingParams : dict, optional
            a dictionary containing the starting values for dynamic parameters for the plant in question, by default
            empty
        """

        super().__init__(params)

    @abc.abstractmethod
    def execute(self, inputs: Dict[str, float]) -> float:
        """This function encapsulates all the calculations a plant needs to do in a single timestep. To do so it will
        call on present and historical inputs if necessary, and will return a single output float. It will also store
        values into whatever fields this plant has that are necessary.

        Parameters
        ----------
        inputs: Dict
            Dict of inputs. Each element in the dictionary should be a single Float. Should by default just have e for
            plants. Depending on the plant's needs, the dict may have to contain more items -- this needs to be
            exception enforced.

        Returns
        -------
        Float
            output value for this plant.
        """
        output = super().execute(inputs)
        return output

    def paramAdj(self, inputs) :
        """Allows for parameter adjustment of specifically allowed parameters/class fields after class has been
           instantiated. Not an abstract method here because not mandatory to be implemented in this child class.

        Parameters
        ----------
        inputs :
            Inputs can be in any format and should conform to the specifics of the actor.
        """
        pass

    @abc.abstractclassmethod
    def getAuxiliaryInfo(self) -> Dict : 
        """Returns a Dict of all parameters and other information in the plant that isn't an input or output.

        Returns
        -------
        Dict
            Dict for all 'internal' information of the plant.
        """
        pass

class controllerBase(actorBase) :
    """Controller base class, child of actorBase class.

    Parent : actorBase
    """
    
    @abc.abstractclassmethod
    def __init__(self, params={}):
        """Initiate plant specific fields here

        Parameters
        ----------
        staticParams : dict, optional
            a dictionary of static parameters for the controller in question, by default empty
        startingParams : dict, optional
            a dictionary containing the starting values for dynamic parameters for the controller in question, by
            default empty
        """

        super().__init__(params)

    @abc.abstractclassmethod
    def execute(self, inputs: Dict[str, float]) -> float:
        """This function encapsulates all the calculations a controller needs to do in a single timestep. To do so it
        will call on present and historical inputs if necessary, and will return a single output float. It will also
        store values into whatever fields this controller has that are necessary.

        Parameters
        ----------
        inputs: Dict
            Dict of inputs. Each element in the dictionary should be a single Float. Should by default just have e for
            controllers. Depending on the controller's needs, the dict may have to contain more items -- this needs to
            be exception enforced.

        Returns
        -------
        Float
            output value for this controller
        """
        output = super().execute(inputs)
        return output

    def paramAdj(self, inputs) :
        """Allows for parameter adjustment of specifically allowed parameters/class fields after class has been
           instantiated. Not an abstract method here because not mandatory to be implemented in this child class.

        Parameters
        ----------
        inputs :
            Inputs can be in any format and should conform to the specifics of the actor.
        """
        pass

    @abc.abstractclassmethod
    def getAuxiliaryInfo(self) -> Dict : 
        """Returns a Dict of all parameters and other information in the controller that isn't an input or output.

        Returns
        -------
        Dict
            Dict for all 'internal' information of the controller.
        """
        pass
    
