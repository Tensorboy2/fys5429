#!/bin/bash
xterm -hold -e "python3 data_pipeline.py 0 100" &
xterm -hold -e "python3 data_pipeline.py 100 200" &
xterm -hold -e "python3 data_pipeline.py 200 300" &
xterm -hold -e "python3 data_pipeline.py 300 400" &
xterm -hold -e "python3 data_pipeline.py 400 500" &
xterm -hold -e "python3 data_pipeline.py 500 600" &
xterm -hold -e "python3 data_pipeline.py 600 700" &
xterm -hold -e "python3 data_pipeline.py 700 800" &
xterm -hold -e "python3 data_pipeline.py 800 900" &
xterm -hold -e "python3 data_pipeline.py 900 1000" 
