Modules to install prior to use:

pip install argparse  
pip install os
pip install datetime
pip install pickle
pip install network
pip install keras
pip install haversine
pip install tk
pip install tkintermapview
pip install functools
pip install sys
pip install re
pip install matplotlib
pip install random
pip install tensorflow
pip install math
pip install scikit-learn

How to train the models and run the pathfinder/gui code.

Plus additional modules assumably already downloaded from the provided Assignment 2A utils.py

to run/output the initial models run: python train_all_models --model model, which can include lstm, gru, rnn. Please use lowercase. 
for example: python train_all_models --model gru

to run pathfinder, which is a gui-less version of our model, run: python pathfinder.py --origin SCATSITE --destination SCATSITE --time 24h time --model model
for example: python pathfinder.py --origin 970 --destination 4273 --time 08:00 --model rnn

the gui can simply be run via: python gui.py
For this
