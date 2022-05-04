"""This is a throw-away script to generate arbitrary r_input tests. Leaving this here so I don't have to keep writing
boilerplate code over and over again everytime I want to make a new arbitrary waveform.
"""

import math
import os
import random
from tqdm import tqdm

# the lines below generate random stepped input file
length = 10**7
stepDuration_min = 10
stepDuration_max = 200
stepLevel_min = -100
stepLevel_max = 100

lines = []
lines.append(f"# This is {length} long series of steps where each step level and the duration of each step is randomized.")

count = 0
duration = random.randint(stepDuration_min, stepDuration_max)
value = random.randint(stepLevel_min, stepLevel_max)
print(f"Generating {length} long random input sequence.")
for i in tqdm(range(length)) :
    if count == duration :
        duration = random.randint(stepDuration_min, stepDuration_max)
        value = random.randint(stepLevel_min, stepLevel_max)
        count = 0
        lines.append(value)
    else :
        lines.append(value)
    count += 1

print(f"Writing the generated input sequence to file.")
file = open('./code/r_inputs/randomStep.csv', 'w')
for line in tqdm(lines):
    file.write(str(line)+'\n')
file.close()
##################################

# # the lines below renames and normalizes to [-100,100] the car derived test files
# for csv in os.listdir('code/r_inputs/') :
#     fullpath = 'code/r_inputs/' + csv

#     if 'carDerivedTest' in csv and 'raw' in csv :
        
#         file = open(fullpath, 'r')

#         header = ''
#         lines = []
#         for line in file :
#             if line[0] == '#' :
#                 header = line[:-1] + ' The values in this file have been normalized to the range [-100,100] from the associated raw input variant.'
#             else :
#                 lines.append(float(line))
#         file.close()

#         in_min = min(lines)
#         in_max = max(lines)
#         out_min = -100
#         out_max = 100
#         newLines = []
#         for line in lines :
#             newLines.append((line - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
        
#         file = open(fullpath[:-8] + '_normalized.csv', 'w')
#         firstLineFlag = True
#         for line in newLines :
#             if firstLineFlag :
#                 file.write(header+'\n')
#                 firstLineFlag = False
#             else :
#                 file.write(str(line)+'\n')
#         file.close()
# ##############################