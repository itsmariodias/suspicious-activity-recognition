# Suspicious Activity Recognition
Action Recognition Model to detect Suspicious activties through Surveillance videos. Developed for the project "Armed, Injured and other Suspicious Activity Recognition using Drone Surveillance"[^note] as part of the internship at Bennet University ([Leadingindia.ai](https://leadingindia.ai/)).

### Dataset used :
 https://www.kaggle.com/mateohervas/dcsass-dataset   
 Delete the second DCSASS Dataset folder and Labels folder

### Requirements :
  `Python 3.10`  
  `ffmpeg`  
  `opencv-python==4.6.0.66`  
  `keras==2.8.0`  
  `numpy==1.24.2`  
  `pandas==1.4.1`  
  `matplotlib==3.5.1`
  `scikit-learn==1.0.2`
  `tensorflow==2.8.0`

### Dataset Preprocessing :
  1. First extract the dataset folder to the same folder as the repository
  2. The `utils` folder contains 3 python scripts to run
  3. run `python utilties/resort_dataset.py`
  4. run `python utilties/create_normal_class.py`
  5. run `python utilties/preprocess.py`
  
### Training :
  The `train_set.ipynb` contains all functions needed to train model. Run all cells and adjust parameters in `main()` function for training.  
  For our experiment we used slowfast model with batch size `8`, img_size `224` and frames `25`.  
  Our slowfast model trained on above settings for `100` epochs can be found here https://www.mediafire.com/file/8cupxxdo9kk936c/slowfast_finalmodel.hdf5/file.
 
### Testing :
  1. First load model or use existing model after training.
  2. Run `predictions()` giving the video to predict as input. We have provided some sample videos in `test/`. You can also directly run `prediction.py`, after manually changing input variable paths.

## References :  
3DCNN : https://github.com/dipakkr/3d-cnn-action-recognition   
Slowfast : https://github.com/facebookresearch/SlowFast  
Keras - Slowfast : https://github.com/xuzheyuan624/slowfast-keras  
Keras - i3D - https://github.com/dlpbc/keras-kinetics-i3d  

[^note]: This is the maintained and original repository for the mirror located [here](https://github.com/LeadingIndiaAI/Armed-Injured-and-other-Suspicious-Activity-Recognition-using-Drone-Surveillance).
