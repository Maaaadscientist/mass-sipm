# Instructions on how to run the macros here
## General features
This directory contains scripts and codes for miscellanious purposes, including
- `harvest.sh` : an executable script for harvesting jobs sharing the same source as major analysis, requiring to loop over those source files without the need to submit jobs on the grid.
- `timeline.py` : a standalone script to extract the timing of both types of the runs from the database, and to align them in a timeline.
- `get_light_logs.py` : a standalone script to extract decoder information of the motor movement from the databased.
- `convert_decoder.py` : a script to convert the log files output from the above `getLightLogs.py` script to suit to the defined coordinates.
- `convert_decoders.sh` : a bash script to perform the feature of `convert_decoder.py` iteratively.
- `new_coordinates.py` : a python script for the coordinates correction, taking the decoder, expected and matched postions and giving the new ones following certain logic.
- `merge_all_csv.py` : a script to merge all csv files after deriving all corrected positions from both the light run and the main run, and also to compare with the timeline and add the time column
- `mu_time_relation.py` : a script to perform the linear regression between reference mu values and LED signal time and extract the parameters to describe the light field at each position.
- `produce_light_map.py` : a script to take the previous results and also the actual design to produce corrected light maps.
- `getReffMu.cc` : compile with `g++ getReffMu.cc $(root-config --cflags --libs) -o getReffMu`, taking an argument of root file path (a root file from the main-match or light-match that contains a TH1F of 16 reference mu in 16 bins).

## How to use
To avoid messing up with such dazzled bash and Python scripts, I will guide you step by step.  
### Major purpose
All of the scripts, text files and root files in this directory aim at an accurate light field matching and measuring. 
### Analysis workflow
To better show this complicated procedure, I have prepared a flow chart:  
![alt text](https://code.ihep.ac.cn/wanghanwen/sipm-massive/-/raw/main/test/illustration.png "A nice illustration of our light field analysis workflow") 
Details will be coming soon!


