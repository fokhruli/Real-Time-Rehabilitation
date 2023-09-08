from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import threading
import ctypes
import _ctypes
import pygame
import sys
from queue import Queue

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from data_processing import Data_Loader
import data_processing
from data_processing import Test_Data_Loader
from graph import Graph
from sgcn_lstm import Stgcn_Lstm
#from stgcn import Stgcn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import PyKinectBodyGame
random_seed = 42  # for reproducibility


import csv
if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# colors for drawing different bodies 
SKELETON_COLORS = [pygame.color.THECOLORS["red"], 
                  pygame.color.THECOLORS["blue"], 
                  pygame.color.THECOLORS["green"], 
                  pygame.color.THECOLORS["orange"], 
                  pygame.color.THECOLORS["purple"], 
                  pygame.color.THECOLORS["yellow"], 
                  pygame.color.THECOLORS["violet"]]


class BodyGameRuntime(object):
    def __init__(self):
        pygame.init()
        self.joints_list=[]
        self.data_queue=Queue()
        self.pred =0
        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1), 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Arial", 50)

        pygame.display.set_caption("Kinect for Windows v2 Body Game")

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data 
        self._bodies = None

    def update_fps(self):
        #fps = str(int(self.clock.get_fps()))
        fps = "Your score : " + str(round(self.pred*100,2))
        fps_text = self.font.render(fps, 1, pygame.Color("red"))
        return fps_text
    def subject(self):
        #fps = str(int(self.clock.get_fps()))
        fps = "Patient"
        fps_text = self.font.render(fps, 1, pygame.Color("red"))
        return fps_text
    
    def ex_name(self):
        #fps = str(int(self.clock.get_fps()))
        fps ="Exercise: Squating"
        fps_text = self.font.render(fps, 1, pygame.Color("red"))
        return fps_text
    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good 
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)

        try:
            pygame.draw.line(self._frame_surface, color, start, end, 8)
        except: # need to catch it due to possible invalid positions (with inf)
            pass

    def draw_body(self, joints, jointPoints, color):
        # Torso
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft);
    
        # Right Arm    
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight);

        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft);

        # Right Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight);

        # Left Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft);


    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def run(self):
        # -------- Main Program Loop -----------
        joints_list=[]
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                    
            # --- Game logic should go here

            # --- Getting frames and drawing  
            # --- Woohoo! We've got a color frame! Let's fill out back buffer surface with frame's data 
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(frame, self._frame_surface)
                frame = None

            # --- Cool! We have a body frame, so can get skeletons
            if self._kinect.has_new_body_frame(): 
                self._bodies = self._kinect.get_last_body_frame()

            # --- draw skeletons to _frame_surface
            if self._bodies is not None: 
                for i in range(0, self._kinect.max_body_count):
                    body = self._bodies.bodies[i]
                    if not body.is_tracked: 
                        continue 

                    joints = body.joints 
                    # convert joint coordinates to color space
                    # print(PyKinectV2.JointType_Count)
                    joint_list=[]
                    for j in range(0, PyKinectV2.JointType_Count):
                        #print(j.Position.x)
                        joint_list.append(joints[j].Position.x)
                        joint_list.append(joints[j].Position.y)
                        joint_list.append(joints[j].Position.z)
                        #joint_list.append([joints[j].Position.x,joints[j].Position.y,joints[j].Position.z])
                    #print(joint_list)
                    joints_list.append(joint_list)
                    if len(joints_list)==105:
                        self.data_queue.put(joints_list)
                        import csv
                        '''
                        with open("F:\swakshar\kinect real time\Test_ex5\Train_X.csv", "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerows(joints_list)
                            # print(joints_list)
                            print("Completed..")'''
                        
                        del joints_list[0]
                    joint_points = self._kinect.body_joints_to_color_space(joints)
                    #depth_point =  self._kinect.body_joints_to_depth_space(joints)
                    self.draw_body(joints, joint_points, SKELETON_COLORS[i])
                    #depth_point = depth_point
                    #print(depth_point)
            
            # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
            # --- (screen size may be different from Kinect's color frame size) 
            h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            target_height = int(h_to_w * self._screen.get_width())
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height));
            self._screen.blit(surface_to_draw, (0,0))
            
            surface_to_draw = None
            self._screen.blit(self.update_fps(), (550,10))
            self._screen.blit(self.ex_name(), (10,10))
            self._screen.blit(self.subject(), (1100,10))
            pygame.display.update()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            self._clock.tick(30)
            #self.hundred_data_fetch()

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()
    
    
    def hundred_data_fetch(self):
        if self.data_queue.empty()==False:
            #print(self.data_queue.get())
            return self.data_queue.get()
        else:
            return 0
            #print("Empty")
            


def data_fetching(game_ob):
    return game_ob.hundred_data_fetch()
    

__main__ = "Kinect v2 Body Game"

game = BodyGameRuntime()

def call(game):

    game.run()
    
    



data_loader = Data_Loader("Kimore ex5")
graph = Graph(len(data_loader.body_part))
train_x, valid_x, train_y, valid_y = train_test_split(data_loader.scaled_x, data_loader.scaled_y, test_size=0.2, random_state = random_seed)
print("Training instances: ", len(train_x))
print("Validation instances: ", len(valid_x))
algorithm = Stgcn_Lstm(train_x, train_y, valid_x, valid_y, graph.AD, graph.AD2, epoach = 1000)


#test_data_loader = Test_Data_Loader("Test_ex5")
model = algorithm.build_model()
model.load_weights("best model/best_model_ex5.hdf5")

t = threading.Thread(target=call, args=(game, ))
t.setDaemon(True)
t.start()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    data= data_fetching(game)
    if data !=0:
        data = np.array(data)
        #print(data)
        predictions = []
        #print(data.shape)
        test_data_loader = Test_Data_Loader(data)
        #print(test_data_loader.scaled_x)
        for i in range(test_data_loader.scaled_x.shape[0]):
            #print('going')            
            prediction = model.predict(test_data_loader.scaled_x[i].reshape(1,test_data_loader.scaled_x[i].shape[0],test_data_loader.scaled_x[i].shape[1],test_data_loader.scaled_x[i].shape[2]))
            predictions.append(prediction[0,0])
        
            prediction = predictions[-1]
            game.pred= prediction
            print(predictions)
    pygame.time.delay(100)

'''joint,joint_list = game.run()
import numpy as np
print(joint_list)
joint_array= np.array(joint_list)
print(joint_array.shape)'''

