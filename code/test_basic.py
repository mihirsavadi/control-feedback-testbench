"""
Testing script for simple PID to FOPDT circuit

Mihir Savadi 29 March 2022
"""

from lib import *
from control_feedback_testbench import scheduler, circuits

if __name__ == "__main__" :

    runObj = scheduler.scheduler(
                type='text',
                source='code/r_inputs/randomStep.csv',
                circuit=circuits.PID_static_FOPDT(
                            P=2,
                            I=0.1,
                            D=0.2,
                            K=2.25,
                            tau=60.5,
                            theta=9.9
                        )
            )
    runObj.update()
    runObj.genPlot(printPlot=True,plotXLimit=(0, 1000))