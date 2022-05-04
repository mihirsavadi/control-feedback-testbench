# Mihir Savadi 29 March 2022

from multiprocessing.sharedctypes import Value
from lib import *
from . import circuits

import re

min_r_input_length = 10

class scheduler() :
    """Scheduler class. Center of this entire application. Contains main program loop, sets up all actors, polls
    them in order, and feeds information between actors depending on circuit being used.
    """

    def __init__(self, 
                 type: str = 'text', 
                 source: str = 'code/r_inputs/sinusoidTest.csv',
                 # circuit object must be initialized either in the constructor argument here or before being passed
                 # into it. The example below is the default - a simple slightly shittily tuned PID to FOPDT pair just
                 # for the sake of behavior demonstration.
                 circuit: circuits.circuitBase = None
                ):
        """Constructor.

        Parameters
        ----------
        type : str, optional
            By default 'text', if r_inputs are to be taken from a text file. Note that in this case, the input for each
            time step must be in a new line. '#' can be used to insert comments. 
            
            Another option is 'instantanous', in which case when the self.update() is called, the new r_value must be
            given as an input.
        source : str
            The source of the text file only if the type argument is 'text'.
        circuit: code.scheduler.circuits.circuitBase
            the circuit that will be initialized and run at every time step.

        """
        # Class fields here
        self.source = source

        self._r_idx_count = 0
        self.r = []
        self.type = type
        self.circuit = circuit

        # Constructor code here
        if type == 'text' :
            self.r = self._parseInput(self.source, min_r_input_length)

        elif type == 'instantaneous':
            pass

        else :
            raise ValueError(f"Scheduler class was instantiated with type '{type}' which is invalid. Can only be 'text' or 'instantaneous'.")

    def update(self, input: float = None, runAll=True):
        """Updates the state and outputs of each actor in the circuit.

        Parameters
        ----------
        input : float, optional
            Must be used if self.type is 'instantanous' to feed in next r value. If self.type is 'text' then this must
            not be used.
        runAll : bool, optional
            Only valid if self.type is 'text'. If True, will run through entire remaining r_input and fill up self.e,u,
            and y. If False, will only update one time step every time this function is called. By default false.

        Returns
        -------
        bool or int
            If scheduler type is 'text' then True if there are more an r values left, False if no r value left. If type
            is 'instantaneous' then returns 0.
        """

        if self.type == 'text' :

            if input != None :
                raise ValueError(f"scheduler class update function must have argument 'input=None'. Right now it is \
                    {input} which is invalid.")
            
            if runAll :
                print(f'Running all r_inputs from {self.source}, from index {self._r_idx_count} till the end...')
                for r_val_idx in tqdm(range(self._r_idx_count, len(self.r))) :
                    self.circuit.execute(self.r[r_val_idx])
                    self._r_idx_count += 1
                return False
            else :
                if self._r_idx_count == len(self.r) :
                    return False
                else :
                    self.circuit.execute(self.r[self._r_idx_count])
                    self._r_idx_count += 1
                    return True
        
        if self.type == 'instantaneous' :
            
            if input == None or type(input) != float or type(input) != int:
                raise ValueError(f"scheduler class update function must have argument 'input' be some float. Right now \
                    it is '{type(input)}' which is invalid.")

            self.r.append(input)
            self.circuit.execute(input)
            self._r_idx_count += 1

            return 0

    def inputChange(self, type: str, source: str = None) :
        """Note that this function influences the use of the 'update' function significantly! Allows the user to change 
        the self.type of this class and/or change the input source if self.type is 'text'. If a new source is added, 
        i.e. the argument is not 'None' (which means the same source used before can be passed
        and it will still trigger the same effect), self._r_idx_count will be reset to 0, and 'self.r' will be
        repopulated. If the user does not wish to change the type then this argument should be set to 'self.type' when
        the function is being called, where 'self' is just whatever the scheduler class object is called.

        Parameters
        ----------
        type : str
            Must be either 'text' or 'instantaneous', otherwise ValueError exception will be thrown.
        source : str, optional
            the path to the input text file that will be used as the new r[t] input stream, by default None
        """
        # check if type is valid
        if type != 'text' or type != 'instantaneous' :
            raise ValueError(f"Scheduler class 'inputReset' functional called with type '{type}' which is invalid. Can only be 'text' or 'instantaneous'.")

        # if type is valid then update class type
        self.type = type

        # then check if type and source argument combo is valid as per function description.
        if type == 'instantaneous' and type(source) != None :
            raise ValueError(f"Scheduler class 'inputReset' functional called with type '{type}', and a source path '{source}' which is not of type 'None', which is invalid. A source can be entered only if type is 'text'.")

        # if all the checks above are sound, go ahead and assign self.r and reset self._r_idx_count
        self.r = self._parseInput(self.source, min_r_input_length)
        self._r_idx_count = 0
        

    def getCircuitInfo(self) -> Dict :
        """Simple getter function to get the self.circuit.getInfo() dictionary from the circuit being implemented. Makes
        it easy for the user to modify the way in which they want to product plots, e.g. in some kind of gui.

        Returns
        -------
        Dict
            self.circuit.getInfo()
        """
        return self.circuit.returnDict

    def genPlot(self, 
                printPlot=False, 
                destination=f'./code/testDump/output_plot_{datetime.now().strftime("%Y%b%d%H%M%S")}.jpg',
                plotXLimit: tuple=None) -> plt:
        """Generates a plot by calling self.circuit's generatePlot function.

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
        plt.title(f'Response Plot for {self.circuit} \n using r_input from {self.source}')

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.set_xlabel(f'Time [unitless]')
        ax1.set_ylabel('Everything Else')

        plt.figtext(0.02, 0, self.circuit.getDescriptionString(), wrap=True, horizontalalignment='left', 
                     verticalalignment='top')
        
        circuitInfoDict = self.circuit.returnDict

        timeAxis = list(range(self._r_idx_count))

        colors = plt.rcParams["axes.prop_cycle"]()
        axis2title = ""
        lastKey = ''
        for key in self.circuit.keysToPlot :    
            c = next(colors)["color"]      
            plottableList = circuitInfoDict[key].copy()
            for i, element in enumerate(plottableList) :
                if type(element) == torch.Tensor :
                    plottableList[i] = float(element)
            plottableList = np.array(plottableList)
            plottableList[np.isnan(plottableList)] = 0
            plottableList[np.isinf(plottableList)] = 0

            if max(plottableList) > 500 or min(plottableList) < -500 :
                axis2title += f"'{key}', "
                ax2.set_ylabel(axis2title + " plots.")
                ax2.plot(timeAxis, plottableList, label=key, color=c)
            else :
                ax1.plot(timeAxis, plottableList, label=key, color=c)
            lastKey = key

        ax1.plot(timeAxis, [0]*len(circuitInfoDict[lastKey]), 'k--', linewidth=0.5) # create a zero axis

        # # uncomment following in order to limit plot range to not deviate too far from r input max min vals
        # yMargin_Lower = int(0.5*min(self.r) - min(self.r)) yMargin_Upper = int(0.5*max(self.r) + max(self.r))
        # plt.ylim([yMargin_Lower, yMargin_Upper])     

        if plotXLimit != None :
            if type(plotXLimit) != tuple :
                raise ValueError(f"Argument 'plotXLimit' in 'genPlot' function of 'scheduler' class must be tuple. Right now it is {type(plotXLimit)} which is invalid!")
            ax1.set_xlim([plotXLimit[0], plotXLimit[1]])

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        if printPlot:
            print(f"Generating plot for {self.source} at {destination}...")
            plt.savefig(destination, bbox_inches='tight', dpi=100)
            print(f"Successfully generated plot!")

        return plt.figure()

    def _parseInput(self, source: str, minLength: int = 10) -> list:
        """This function simply parses an input text file intended to be used as r[t] inputs in a circuit, and outputs a
        list of floats containing said information, whilst ignoring '#' comments. Will throw an error if text file
        number of values is less than 'minLength' or is 0.

        Parameters
        ----------
        source : str
            the path to the input text file

        Returns
        -------
        list
            A list of floats parsed from the text file pointed to by the path in 'source'.
        """
        if len(source) == 0 :
            raise ValueError('r_value source text file is empty')

        file = open(source, 'r')
        
        print(f"Parsing input file '{source}'!")
        outputList = []
        for line in tqdm(file.readlines()):
            line = re.sub('#.*', '', line)[:-1]
            if len(line) != 0 :
                outputList.append(float(line))

        if len(outputList) < minLength :
            raise Exception(f'r_input size is too short at {len(outputList)}. Needs to be at least {minLength}')

        return outputList