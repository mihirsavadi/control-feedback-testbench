"""
Testing/demonstration script for RTCFL with feedforward RNN actors. Also performs poorly. Maybe I just haven't tuned it
well enough. Still way less out of hand than the RTCFL-DNN though.

Mihir Savadi 2 May 2022
"""

from pdb import runcall
from lib import *

from control_feedback_testbench import base, plants, controllers, circuits, scheduler


if __name__ == "__main__" :
    torch.autograd.set_detect_anomaly(True)

    # first testing the purely feedforward based NN. Garbage performance.
    runObj = scheduler.scheduler(
                type='text',
                source='code/r_inputs/randomStep.csv',
                circuit=circuits.RTCFL_RNN(plant_bb_model=plants.FOPDT(params={'K' : 2.25, 'tau': 60.5, 'theta' : 9.9}),
                                           plant_nn_stackCount             = 2,
                                           plant_nn_learningRate           = 0.01,
                                           plant_nn_trainSeqLen            = 5,
                                           plant_nn_trainSeqAddPeriod      = 5,
                                           plant_nn_trainSeqDBSize         = 100,
                                           plant_nn_minLoss                = 150000,
                                           controller_nn_stackCount        = 1,
                                           controller_nn_learningRate      = 0.01,
                                           controller_nn_trainSeqLen       = 50,
                                           controller_nn_trainSeqAddPeriod = 50,
                                           controller_nn_trainSeqDBSize    = 100,
                                           controller_nn_minLoss           = 3,
                                           device='cpu' # runs way faster on cpu for some reason
                )
    )
    # count = 1
    # errorMem = [0.0]*500
    # while runObj.update(runAll=False) :
    #     error = runObj.getCircuitInfo()['e'][-1]
    #     errorMem.pop(0)
    #     errorMem.append(error)
    #     average = sum(errorMem)/len(errorMem)
    #     print(f"timestep: {count}, mode: {runObj.getCircuitInfo()['mode'][-1]},\tr[t] = {runObj.getCircuitInfo()['r'][-1]},\ty[t] = {round(runObj.getCircuitInfo()['y'][-1], 2)},\tlabel = {round(runObj.getCircuitInfo()['label'][-1], 2)},\terror = {round(error, 2)},\taverage error = {round(average, 2)},\tloss_plnt = {runObj.circuit.returnDict['loss_plnt'][-1]},\tu[t] = {round(runObj.getCircuitInfo()['u'][-1].item(), 2)}")
    #     count += 1

    totalIterations = 5400
    print(f'Running {totalIterations} iterations of the input.')
    for i in tqdm(range(totalIterations)) :
        if runObj.update(runAll=False) == False :
            break
    runObj.genPlot(printPlot=True)