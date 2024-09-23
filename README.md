# Performance studies of the gnn pipeline

To use this repo, you must first have installed the gnn pipeline. 
This will run in the same virtual environment, but you need plotly to run the plots.

The notebook `Timing inference` loads pre-trained checkpoints. 
It then uses the basicTimer class to setup the timer. This might needs slight tweaking
and there are explicit paths to data. It writes out files of the pytorch timings that can 
be read as txt files or through profilers such as chrome. 

The data can be loaded from point clouds or from raw data. 

