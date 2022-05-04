"""
Testing/demonstration script for RTCFL with feedforward DNN actors. Performs poorly.

Mihir Savadi 2 May 2022
"""

from pdb import runcall
from lib import *

from control_feedback_testbench import base, plants, controllers, circuits, scheduler


if __name__ == "__main__" :

    # first testing the purely feedforward based NN. Garbage performance.
    runObj = scheduler.scheduler(
                type='text',
                source='code/r_inputs/randomStep.csv',
                circuit=circuits.RTCFL_DNN(
                    plant_bb      = plants.FOPDT(params={'K' : 2.25, 'tau': 60.5, 'theta' : 9.9}),
                    plant_nn      = plants.DNN(name="plant_nn", 
                                               params={'n'                   : 500,
                                                       'hidden layers'       : [750, 750],
                                                       'activation function' : torch.nn.ReLU,
                                                       'learning rate'       : 0.001,
                                                       'device'              : 'cpu'
                                                       }),
                    plant_minError = 5,
                    plant_k        = 1,
                    plant_l        = 1000,
                    controller_nn  = controllers.DNN(name="plant_nn", 
                                                    params={'n'                   : 500,
                                                            'hidden layers'       : [750, 750],
                                                            'activation function' : torch.nn.ReLU,
                                                            'learning rate'       : 0.001,
                                                            'device'              : 'cpu'
                                                           }),
                    controller_minLoss = 2000,
                    controller_k       = 1,
                    controller_l       = 1000
                )
    )
    # errorMem = [0.0]*500
    # while runObj.update(runAll=False) :
    #     error = runObj.getCircuitInfo()['e'][-1]
    #     errorMem.pop(0)
    #     errorMem.append(error)
    #     average = sum(errorMem)/len(errorMem)
    #     print(f"mode: {runObj.getCircuitInfo()['mode'][-1]},\tr[t] = {runObj.getCircuitInfo()['r'][-1]},\tu[t] = {round(runObj.getCircuitInfo()['y'][-1], 2)},\terror = {round(error, 2)},\taverage error = {round(average, 2)}")

    totalIterations = 10000
    print(f'Running {totalIterations} iterations of the input.')
    for i in tqdm(range(totalIterations)) :
        if runObj.update(runAll=False) == False :
            break
    runObj.genPlot(printPlot=True)