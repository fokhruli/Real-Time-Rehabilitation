# Real-Time-Rehabilitation-Interface-with-kinectv2
Real time rehabilitation assessment with kinect v2. We implement the spatio temporal graph convolutional neural network for rehabilitation exercise assessment according to the following paper:  S. Deb, M. F. Islam, S. Rahman and S. Rahman, "[Graph Convolutional Networks for Assessment of Physical Rehabilitation Exercises](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9709340)," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 30, pp. 410-419, 2022, doi: 10.1109/TNSRE.2022.3150392.

# How to run this code
First you need the Microsoft Kinect v2 and install the associated Kinect SDK. After that follow these steps:
1) Install the requirements.txt file the command `pip install -r requirements.txt`.
2) Download the model parameters from the [google drive](https://drive.google.com/drive/u/1/folders/1c2Nucl8iIFhDvPZUjdkTFYoksaX1TpK_) and store them in the best model folder.
3) Run the `PyKinectBodyGame_v1.py` file with the command `python PyKinectBodyGame_v1.py` in the terminal.
 
# Prediction
Output for the healthy person. The correctness score is shown in real time at the top middle of the screen. Consequently, exercise name is shown at the top left.

https://github.com/SwaksharDeb/Real-Time-Rehabilitation/assets/49334830/a5dcd42b-e1fa-4ea4-902a-f7c19e5dd205



https://github.com/SwaksharDeb/Real-Time-Rehabilitation/assets/49334830/31c4520c-9886-4c9d-b7f2-0f31254387d8

Output for the patient. The correctness score is shown in real time at the top middle of the screen. Consequently, exercise name is shown at the top right.


https://github.com/SwaksharDeb/Real-Time-Rehabilitation/assets/49334830/46edbbff-5924-47e0-a190-89c740976a8e



https://github.com/SwaksharDeb/Real-Time-Rehabilitation/assets/49334830/b5153988-c712-484c-9b68-34578a748fbb

