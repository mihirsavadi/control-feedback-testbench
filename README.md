# Control-Feedback Testbench (CFT)

[![License: CC BY-NC-SA
4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This repository contains the [source code](./code/), and [documentation](./documentation/) for: 
1. **The Control-Feedback Testbench (CFT)**
2. **The experimental Real-Time Convergent Feedback Loop (RTCFL)**

The CFT presents a modular and scalable control-feedback loop simulation framework in python. It can be used to
implement and evaluate, with ease, arbitrary controllers, plants (and other actors), in arbitrary circuits with
arbitrary stimuli in a simulated real-time environment. Source code pertaining to it is located in
[code/control_feedback_testbench](code/control_feedback_testbench).

The RTCFL is a novel and experimental self-convergent SISO controller, based on machine learning principles and neural
networks - sort of like a very generalizable auto-tuning PID controller. The ML framework of choice in this project is
Pytorch, which is exclusively used. Software implementations can be seen in the class definitions located in
[code/control_feedback_testbench/circuits.py](code/control_feedback_testbench/circuits.py), and the test files:
[code/test_rtcfl-DNN.py](code/test_rtcfl-DNN.py) and [code/test_rtcfl-RNN.py](code/test_rtcfl-RNN.py).

A good, overall, up-to-date explanation of everything can be read in the document located at
[documentation/VT_ECE4424_deliverables/final_report/finalReport_v0.2/finalreport_mihirsavadi1_v0.2.pdf](documentation/VT_ECE4424_deliverables/final_report/finalReport_v0.2/finalreport_mihirsavadi1_v0.2.pdf).

**An Important Series of Related Notes:**
1. This repository and all its contents pertain to my deliverables for the Final Project of the 'ECE 4424 Machine
  Learning' class I took in the Spring of 2022 at Virginia Tech, taught by [Prof. Debswapna
  Bhattacharya](https://cs.vt.edu/people/faculty/deb-bhattacharya.html).
2. As such, for a comprehensive dive into what this is all about, please see the readme's in the [code/](./code/) and
  [documentation/VT_ECE4424_deliverables/](./documentation/VT_ECE4424_deliverables/) directories (located
  [here](./code/README.md) and [here](./documentation/VT_ECE4424_deliverables/README.md) respectively), and explore the
  explanations and links to resources contained within them.
3. At the time of creating this repository and completing the ECE4424 final project, the RTCFL was far from proven as a
  viable solution. However its promise warrants further work and research into it.
4. Likewise, the scope of the CFT's utility goes far beyond the RTCFL project - hopefully it will be utilized as such.