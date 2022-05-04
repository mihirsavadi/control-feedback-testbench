# Control-Feedback Testbench (CFT) Source Code

[![License: CC BY-NC-SA
4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

*NOTE: Please see [the paper
here](../../documentation/VT_ECE4424_deliverables/final_report/finalReport_v0.2/finalreport_mihirsavadi1_v0.2.pdf) for
greater context of the purpose of the CFT - section 4 is paraphrased [below](#purpose).* 

*NOTE: For a quick description of the CFT source code outline, [jump to here](#cft-module-code-overview).*

## Purpose

In order to efficiently build, analyze, and maintain a variety of control-feedback circuit designs without compromising
design flexibility and granularity, the use of a modular 'testbench' environment/platform was imperative. Such a
testbench, simply referred to as the 'Control-Feedback Testbench' (or CFT), was built.

The testbench implemented is fundamentally modular, employing object orientated design with heavy use of both
inheritance (to enforce inter-class communication consistency without compromising modularity) and composition. At its
core, the testbench is based on the 'actor' paradigm, whereby each element in any given circuit is treated as a black
box, or an actor, whose basic inputs and outputs are standardized by an abstract actor base class, and whose internal
calculations are abstracted away. Actors can be controllers, plants, or any other influencing element of a circuit. 

Actor objects are then instantiated inside a circuit class, which contain standardized inputs and outputs themselves,
and abstract away the interconnections (whether chronological or parallel) between whatever actors may be present in
said particular circuit. These standardizations are again enforced by an abstract circuit base class. Circuit classes
contain a single function to update actor outputs according to its defined interconnections for each time-step, as well
as store historical data for each 'wire' in the circuit.

Finally, a scheduler class is what the user interacts with in order to utilize the testbench -- it contains an instance
of a particular circuit class, whose parameters are determined on initialization of a scheduler object; it provides a
function to get inputs into the circuit, either by time-step by time-step polling or by parsing an external text file;
it provides a function to probe the circuit to get information on its stored current and historical data; and it
provides a function to generate a convenient plot to quickly visualize circuit behavior.

The user can create arbitrary actor classes (controllers, plants, etc.) and circuit classes, which will function
predictably with the aforementioned scheduler class, as long as the respective abstract base classes are adhered to.
Thus, a user can use the scheduler class of this testbench to automate testing and analysis of a variety of controllers,
plants, and circuits, in an arbitrarily simulated real-time environment.

## CFT Module Code Overview

The codebase for the CFT (in '[control_feedback_testbench/'](control_feedback_testbench/)) is explained below:
- [base.py](control_feedback_testbench/base.py) contains the main actor abstract base classes, as well as the derived
  plant and controller base classes. This must not be altered. All actors implemented in
  [plants.py](control_feedback_testbench/plants.py) and [controllers.py](control_feedback_testbench/controllers.py) must
  inherit these classes (i.e. a plant actor must inherit the plant class and implement all its abstract methods and so
  on). 
- [plants.py](control_feedback_testbench/plants.py) contains all the plant implementations, such as FOPDT, SOPDT, RNN,
  and DNN. More can controllers can be added to here.
- [controllers.py](control_feedback_testbench/controllers.py) contains all the controller implementations, such as PID,
  RNN, and DNN.
- circuits.py contains the circuit base class as well as the various circuit implementations -- e.g. static PID-FOPDT,
  RNN-RTCFL, and DNN-RTCFL. More circuit classes be added here by a user. 
- [scheduler.py](control_feedback_testbench/scheduler.py) composes an instance of a user-chosen
  [circuits.py](control_feedback_testbench/circuits.py) class. This class is what the user interacts with to use the
  'CFT'.

Remaining directories explained below:
- Files contained in [this](.) directory beginning with 'test' contain examples of the CFT being used in practice, e.g.
  [test_basic.py](test_basic.py), [test_rtcfl-DNN.py](test_rtcfl-DNN.py) and [test_rtcfl-RNN.py](test_rtcfl-RNN.py).
- [lib/](lib/) contains a summary of all the python module dependencies.
- [r_inputs/](r_inputs/) contains r[t] input sequences to feed into circuits. 
    - The files starting containing 'car derived' (e.g.
      [r_inputs/carDerivedTest-32_raw.csv](r_inputs/carDerivedTest-32_raw.csv)) are intended to provide a natural/more
      realistic series of time data - it was gathered by logging the OBD2 data from a short journey in my car.
    - There are other input files that are more generic (e.g. [r_inputs/sinusoidTest.csv](r_inputs/sinusoidTest.csv) and
      [r_inputs/simpleStep.csv](r_inputs\simpleStep.csv)).
    - There also exists a python script [r_inputs/easyTestGenerator.py](r_inputs/easyTestGenerator.py) to
      programmatically generate input files, e.g. [r_inputs/randomStep.csv](r_inputs/randomStep.csv).
- [testDump/](testDump/) is where the plots generated by the .genPlot() function of user-created objects of
  [scheduler.py](control_feedback_testbench/scheduler.py) are placed by default. There also contains information from
  previous personal experimentation runs (e.g.
  [testDump/garbageDNNperformanceExample_03May2022_2-32am.txt](testDump/garbageDNNperformanceExample_03May2022_2-32am.txt)).