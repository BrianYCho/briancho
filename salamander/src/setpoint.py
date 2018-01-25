#!/usr/bin/env python

import random
import rospy
import numpy as np
import math
import argparse
import time
import pprint as pp
import tflearn
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from scipy.integrate import odeint
from std_msgs.msg import Int8MultiArray
from std_msgs.msg import Int8
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelConfiguration

t_ = []
EXPLORE = 100000.
epsilon = 1.0
drive_sig = []
done_step = 0
score = 0.0
target_pos = [0,6]
v_body = 0.0
v_limb = 0.0
R_body = 0.0
R_limb = 0.0

theta_body_01 = 0.0
r_body_01 = 0.0
rdot_body_01 = 0.0 
x_body_01 = 0.0
theta_body_02 =0.0
r_body_02 = 0.0 
rdot_body_02 = 0.0
x_body_02 = 0.0
theta_body_03 =0.0 
r_body_03 = 0.0 
rdot_body_03 = 0.0 
x_body_03 = 0.0
theta_body_04 =0.0 
r_body_04 = 0.0 
rdot_body_04 = 0.0
x_body_04 = 0.0
theta_body_05 =0.0 
r_body_05 = 0.0 
rdot_body_05 = 0.0 
x_body_05 = 0.0
theta_body_06 =0.0 
r_body_06 = 0.0 
rdot_body_06 = 0.0 
x_body_06 = 0.0
theta_body_07 =0.0 
r_body_07 = 0.0 
rdot_body_07 = 0.0 
x_body_07 = 0.0
theta_body_08 =0.0 
r_body_08 = 0.0 
rdot_body_08 = 0.0 
x_body_08 = 0.0
theta_body_09 =0.0 
r_body_09 = 0.0 
rdot_body_09 = 0.0 
x_body_09 = 0.0
theta_body_10 =0.0 
r_body_10 = 0.0 
rdot_body_10 = 0.0 
x_body_10 = 0.0
theta_body_11 =0.0 
r_body_11 = 0.0 
rdot_body_11 = 0.0 
x_body_11 = 0.0
theta_body_12 =0.0 
r_body_12 = 0.0 
rdot_body_12 = 0.0 
x_body_12 = 0.0
theta_body_13 =0.0 
r_body_13 = 0.0 
rdot_body_13 = 0.0 
x_body_13 = 0.0
theta_body_14 =0.0 
r_body_14 = 0.0 
rdot_body_14 = 0.0 
x_body_14 = 0.0
theta_body_15 =0.0 
r_body_15 = 0.0 
rdot_body_15 = 0.0 
x_body_15 = 0.0
theta_body_16 =0.0 
r_body_16 = 0.0 
rdot_body_16 = 0.0 
x_body_16 = 0.0

theta_limb_17 =0.0 
r_limb_17 = 0.0 
rdot_limb_17 = 0.0 
x_limb_17 = 0.0
theta_limb_18 =0.0 
r_limb_18 = 0.0 
rdot_limb_18 = 0.0 
x_limb_18 = 0.0
theta_limb_19 =0.0 
r_limb_19 = 0.0 
rdot_limb_19 = 0.0 
x_limb_19 = 0.0
theta_limb_20 =0.0 
r_limb_20 = 0.0 
rdot_limb_20 = 0.0 
x_limb_20 = 0.0
MA_ = []

alpha_1 = 0.5
alpha_2 = 0.6
alpha_3 = 0.7
alpha_4 = 0.8
alpha_5 = 0.9
alpha_6 = 1.0
alpha_7 = 0.5
alpha_8 = 0.7
d_walk = 3.0
d = 3.0
d_swim = 4.0
c_v1_body = 0.2
c_v0_body = 0.3
c_v1_limb_w = 0.2
c_v1_limb_s = 0.0
c_v0_limb_w = 0.0
c_v0_limb_s = 0.0
c_R1_body = 0.065
c_R0_body = 0.196
c_R1_limb_w = 0.131
c_R0_limb_w = 0.131
c_R1_limb_s = 0.0
c_R0_limb_s = 0.0
a = 20.0
W_downbody = 10.0
pi_downbody = -2*np.pi/8
W_upbody = 10.0
pi_upbody = 2*np.pi/8
W_conbody = 10.0
pi_conbody = np.pi
W_limbtobody = 30.0
pi_limbtobody = np.pi
W_withinlimb = 10.0
pi_withinlimb = np.pi

MAX_EPISODE = 1000
n_hidden_1 = 400
n_hidden_2 = 300
n_input = 3
n_output = 26
training_epochs = 200

saver = tf.train.Saver()
model_dir = "tf_model/model_pretrain.ckpt"

env_ground = np.array([[1.,0.,0.]])
env_water = np.array([[0.,1.,0.]])
env_mud = np.array([[0.,0.,1.]])

x_walk = np.array([[1,0,0] for _ in range(1000)])
x_swim = np.array([[0,1,0] for _ in range(1000)])
x = np.concatenate((x_walk,x_swim), axis=0)
pos = np.array([[random.uniform(-1.0, 1.0), random.uniform(-0.5,3.5)] for _ in range(2000)])
x = np.concatenate((x,pos), axis=1)
vel = np.array([[random.uniform(-0.3, 0.3), random.uniform(-0.3,0.3)] for _ in range(2000)])
x = np.concatenate((x,vel), axis =1)
y_walk = np.array([[alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, d_walk, c_v1_body,c_v0_body, c_v1_limb_w, c_v0_limb_w, c_R1_body, c_R0_body, c_R1_limb_w, c_R0_limb_w, a, W_downbody, W_upbody, W_conbody, W_limbtobody, W_withinlimb, pi_downbody,  pi_upbody, pi_conbody, pi_limbtobody, pi_withinlimb] for _ in range(1000)])
y_swim = np.array([[alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, d_swim, c_v1_body,c_v0_body, c_v1_limb_s, c_v0_limb_s, c_R1_body, c_R0_body, c_R1_limb_s, c_R0_limb_s, a, W_downbody, W_upbody, W_conbody, W_limbtobody, W_withinlimb, pi_downbody,  pi_upbody, pi_conbody, pi_limbtobody, pi_withinlimb] for _ in range(1000)])
y = np.concatenate((y_walk, y_swim),axis=0)

rospy.init_node('setpoint')
ros_time = rospy.get_rostime()
rate = rospy.Rate(50)
reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
reset_jointstate = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
pub_limb1 = rospy.Publisher('/rrbot/limb1_position_controller/command',Float64, queue_size=1)
pub_limb2 = rospy.Publisher('/rrbot/limb2_position_controller/command',Float64, queue_size=1)
pub_limb3 = rospy.Publisher('/rrbot/limb3_position_controller/command',Float64, queue_size=1)
pub_limb4 = rospy.Publisher('/rrbot/limb4_position_controller/command',Float64, queue_size=1)
pub_body1 = rospy.Publisher('/rrbot/body1_position_controller/command',Float64, queue_size=1)
pub_body2 = rospy.Publisher('/rrbot/body2_position_controller/command',Float64, queue_size=1)
pub_body3 = rospy.Publisher('/rrbot/body3_position_controller/command',Float64, queue_size=1)
pub_body4 = rospy.Publisher('/rrbot/body4_position_controller/command',Float64, queue_size=1)
pub_body5 = rospy.Publisher('/rrbot/body5_position_controller/command',Float64, queue_size=1)
pub_body6 = rospy.Publisher('/rrbot/body6_position_controller/command',Float64, queue_size=1)


saver = tf.train.Saver()

pose_x = 0.0
pose_y = 0.0
pose_x1 = 0.0
pose_y1= 0.0
pose_x2 = 0.0
pose_y2 = 0.0

pose_x3 = 0.0
pose_y3 = 0.0

pose_x4 = 0.0
pose_y4 = 0.0

pose_x5 = 0.0
pose_y5 = 0.0

pose_x6 = 0.0
pose_y6 = 0.0

pose_x7 = 0.0
pose_y7 = 0.0

pose_x8 = 0.0
pose_y8 = 0.0

vel_x1 = 0.0
vel_y1 = 0.0
vel_x5 = 0.0
vel_y5 = 0.0

def callback(msg):
    global setpointData
    setpointData = msg.position


def callbackodom(msg):
    global pose_x
    global pose_y
    pose_x = msg.pose.pose.position.x
    pose_y = msg.pose.pose.position.y

def callbackodom1(msg):
    global pose_x1
    global pose_y1
    global vel_x1
    global vel_y1
    pose_x1 = msg.pose.pose.position.x
    pose_y1 = msg.pose.pose.position.y
    vel_x1 = msg.twist.twist.linear.x
    vel_y1 = msg.twist.twist.linear.y

def callbackodom2(msg):
    global pose_x2
    global pose_y2
    pose_x2 = msg.pose.pose.position.x
    pose_y2 = msg.pose.pose.position.y

def callbackodom3(msg):
    global pose_x3
    global pose_y3
    pose_x3 = msg.pose.pose.position.x
    pose_y3 = msg.pose.pose.position.y

def callbackodom4(msg):
    global pose_x4
    global pose_y4
    pose_x4 = msg.pose.pose.position.x
    pose_y4 = msg.pose.pose.position.y

def callbackodom5(msg):
    global pose_x5
    global pose_y5
    global vel_x5
    global vel_y5
    pose_x5 = msg.pose.pose.position.x
    pose_y5 = msg.pose.pose.position.y
    vel_x5 = msg.twist.twist.linear.x
    vel_y5 = msg.twist.twist.linear.y

def callbackodom6(msg):
    global pose_x6
    global pose_y6
    pose_x6 = msg.pose.pose.position.x
    pose_y6 = msg.pose.pose.position.y

def callbackodom7(msg):
    global pose_x7
    global pose_y7
    pose_x7 = msg.pose.pose.position.x
    pose_y7 = msg.pose.pose.position.y

def callbackodom8(msg):
    global pose_x8
    global pose_y8
    pose_x8 = msg.pose.pose.position.x
    pose_y8 = msg.pose.pose.position.y



def distance(x1,x2,y1,y2):
    x = x2-x1
    y = y2-y1
    result = math.sqrt(x**2 + y**2)
    return result

sub_state = rospy.Subscriber('/ground_truth/state',Odometry,callbackodom,queue_size=1)
sub_state1 = rospy.Subscriber('/ground_truth/state1',Odometry,callbackodom1,queue_size=1)
sub_state2 = rospy.Subscriber('/ground_truth/state2',Odometry,callbackodom2,queue_size=1)
sub_state3 = rospy.Subscriber('/ground_truth/state3',Odometry,callbackodom3,queue_size=1)
sub_state4 = rospy.Subscriber('/ground_truth/state4',Odometry,callbackodom4,queue_size=1)
sub_state5 = rospy.Subscriber('/ground_truth/state5',Odometry,callbackodom5,queue_size=1)
sub_state6 = rospy.Subscriber('/ground_truth/state6',Odometry,callbackodom6,queue_size=1)
sub_state7 = rospy.Subscriber('/ground_truth/state7',Odometry,callbackodom7,queue_size=1)
sub_state8 = rospy.Subscriber('/ground_truth/state8',Odometry,callbackodom8,queue_size=1)


sub_joint = rospy.Subscriber('/rrbot/joint_states', JointState, callback, queue_size=1)

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.deque.clear()
        self.count = 0

class ActorNetwork(object):
    

    def __init__(self, sess, graph,state_dim, action_dim, learning_rate, tau, batch_size):
        self.sess = sess
        self.graph = graph
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        param_w = [alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, d_walk, c_v1_body,c_v0_body, c_v1_limb_w, c_v0_limb_w, c_R1_body, c_R0_body, c_R1_limb_w, c_R0_limb_w, a, W_downbody, W_upbody, W_conbody, W_limbtobody, W_withinlimb, pi_downbody,  pi_upbody, pi_conbody, pi_limbtobody, pi_withinlimb]
        param_s = [alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, d_swim, c_v1_body,c_v0_body, c_v1_limb_s, c_v0_limb_s, c_R1_body, c_R0_body, c_R1_limb_s, c_R0_limb_s, a, W_downbody, W_upbody, W_conbody, W_limbtobody, W_withinlimb, pi_downbody,  pi_upbody, pi_conbody, pi_limbtobody, pi_withinlimb]
        self.max = np.zeros( [26] )
        self.min = np.zeros( [26] )
        exception = [9,10,13,14]

        for i in range(26):
            if i == 6: continue
            self.max[i] = param_w[i]*(1.05)
            self.min[i] = param_w[i]*(0.95)

        self.max[6] = 5.0
        self.min[6] = 2.0
        for i in range(len(exception)):
            idx = exception[i]
            self.max[idx] = max(param_s[idx], param_w[idx])*(1.05)
            self.min[idx] = min(param_s[idx], param_w[idx])*(0.95)
        

        
        print(self.max)
        print(self.min)
        self.inputs, self.out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        
        self.target_inputs, self.target_out= self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

        self.pre_target = tf.placeholder("float",[None, self.a_dim])
        self.pre_loss = tf.reduce_mean(tf.square(self.pre_target-self.out))
        self.pre_train = tf.train.AdamOptimizer(0.01).minimize(self.pre_loss)
        
        # tf.summary.scalar('Loss', self.pre_loss)
        # self.summary_merge = tf.summary.merge_all()
        # self.writer_pre = tf.summary.FileWriter(args['log_dir'], self.sess.graph) 

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 500)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        #Final layer weights are init to Uniform[-3e-2, 3e-2]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim, weights_init=w_init)
        net1,net2 = tf.split(out,[21,5],1)
        net1_a = tf.nn.sigmoid(net1)
        net2_a = tf.nn.tanh(net2)

        net_a = tf.concat([net1_a,net2_a],axis=1)
        return inputs, net_a

    def pretrain(self, inputs, outputs):
        target=[]
        for i in range(2000):
            target.append(self.normalize(list(outputs[i])))
             #,merge, ,self.summary_merge
        _, loss = self.sess.run([self.pre_train,self.pre_loss], feed_dict={   
            self.inputs: inputs,
            self.pre_target : np.array( target )
        })
        return loss #,merge
    

    def prepredict(self,inputs):
        params = self.sess.run(self.out, feed_dict={self.inputs: inputs})
        return params
    
    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })
    def scaleshift(self, inputs):
        result = np.zeros([26])
        for i in range(26):
            result[i] = (self.max[i]-self.min[i])*inputs[i] + self.min[i]
        return result

    def normalize(self, inputs):
        result = np.zeros([26])
        for i in range(26):
            if (self.max[i]-self.min[i]) == 0:
                result[i] = 0.
            else :
                result[i] = (inputs[i] - self.min[i])/(self.max[i]-self.min[i])
        return result

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        
        self.inputs, self.action, self.out = self.create_critic_network()
        self.network_params = tf.trainable_variables()[num_actor_vars:]
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])


        net_a = tf.concat([inputs,action],axis=1)

        net = tflearn.fully_connected(net_a, 500)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)    
         # net = tflearn.activation(tf.matmul(inputs, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        # net = tflearn.fully_connected(inputs, 600)
        # net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.activations.relu(net)

        # t1 = tflearn.fully_connected(net, 600)
        # t2 = tflearn.fully_connected(action, 600)

        # net = tflearn.activation(
        #     tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

class queue:
	
    def __init__(self, max_size, length, init_cnt=False) :
        self.cnt = 0
        self.que = np.zeros( [max_size, length] )
        self.size = max_size-1
        self.len = max_size
        self.length = length
        self.init_cnt = init_cnt
    
    def clear(self):
        self.cnt = 0

    def data(self):
        return self.que

    def avr(self):
        return sum( self.que ) / self.len 	#np.mean(self.que,axis=0)

    def pop(self):
        result = self.que[0]
        self.que = np.delete( self.que, [0], axis=0 )
        self.que = np.insert( self.que, [self.size], [0 for _ in range(self.length)], axis=0 )
        return result

    def push(self,data):

        self.pop()
        self.que[self.size] = data

        if self.cnt < self.size :
            self.cnt = self.cnt + 1
            return False
        else :
            if self.init_cnt == True:
                self.cnt = 0
        return True

def oscillator_body_01(y,t):
    
    #v1_body,c_v0_body, c_v1_limb, c_v0_limb, c_R1_body, c_R0_body, c_R1_limb, c_R0_limb, a, W_downbody, pi_downbody, W_upbody, pi_upbody, W_conbody, pi_conbody, W_limbtobody, pi_limbtobody, W_withinlimb, pi_withinlimb
    dydt = [2*np.pi*v_body + r_limb_17 * W_limbtobody*np.sin(theta_limb_17-y[0]-pi_limbtobody) + r_body_02 * W_upbody*np.sin(theta_body_02-y[0]-pi_upbody) + r_body_09 * W_conbody*np.sin(theta_body_09-y[0]-pi_conbody) ,
        y[2],
        a*(a/4*(R_body-y[1])-y[2])] 
    return dydt

def oscillator_body_02(y,t):
    dydt = [2*np.pi*v_body + r_limb_17 * W_limbtobody*np.sin(theta_limb_17-y[0]-pi_limbtobody) + r_body_01 * W_downbody*np.sin(theta_body_01-y[0]-pi_downbody) + r_body_03 * W_upbody*np.sin(theta_body_03-y[0]-pi_upbody) + r_body_10 * W_conbody*np.sin(theta_body_10-y[0]-pi_conbody),
        y[2],
        a*(a/4*(R_body-y[1])-y[2])]
    return dydt

def oscillator_body_03(y,t):
    dydt = [2*np.pi*v_body + r_limb_17 * W_limbtobody*np.sin(theta_limb_17-y[0]-pi_limbtobody) + r_body_02 * W_downbody*np.sin(theta_body_02-y[0]-pi_downbody) + r_body_04 * W_upbody*np.sin(theta_body_04-theta_body_03-pi_upbody) + r_body_11 * W_conbody*np.sin(theta_body_11-y[0]-pi_conbody),
        y[2],
        a*(a/4*(R_body-y[1])-y[2])]
    return dydt

def oscillator_body_04(y,t):
    dydt = [2*np.pi*v_body + r_limb_17 * W_limbtobody*np.sin(theta_limb_17-y[0]-pi_limbtobody) + r_body_03 * W_downbody*np.sin(theta_body_03-y[0]-pi_downbody) + r_body_05 * W_upbody*np.sin(theta_body_05-theta_body_04-pi_upbody) + r_body_12 * W_conbody*np.sin(theta_body_12-y[0]-pi_conbody),
        y[2],
        a*(a/4*(R_body-y[1])-y[2])]
    return dydt  

def oscillator_body_05(y,t):
    dydt = [2*np.pi*v_body + r_limb_19 * W_limbtobody*np.sin(theta_limb_19-y[0]-pi_limbtobody) + r_body_04 * W_downbody*np.sin(theta_body_04-y[0]-pi_downbody) + r_body_06 * W_upbody*np.sin(theta_body_06-y[0]-pi_upbody) + r_body_13 * W_conbody*np.sin(theta_body_13-y[0]-pi_conbody),
        y[2],
        a*(a/4*(R_body-y[1])-y[2])]
    return dydt  

def oscillator_body_06(y,t):
    dydt = [2*np.pi*v_body + r_limb_19 * W_limbtobody*np.sin(theta_limb_19-y[0]-pi_limbtobody) + r_body_05 * W_downbody*np.sin(theta_body_05-y[0]-pi_downbody) + r_body_07 * W_upbody*np.sin(theta_body_07-y[0]-pi_upbody) + r_body_14 * W_conbody*np.sin(theta_body_14-y[0]-pi_conbody),
        y[2],
        a*(a/4*(R_body-y[1])-y[2])]
    return dydt  

def oscillator_body_07(y,t):
    dydt = [2*np.pi*v_body + r_limb_19 * W_limbtobody*np.sin(theta_limb_19-y[0]-pi_limbtobody) + r_body_06 * W_downbody*np.sin(theta_body_06-y[0]-pi_downbody) + r_body_08 * W_upbody*np.sin(theta_body_08-y[0]-pi_upbody) + r_body_15 * W_conbody*np.sin(theta_body_15-y[0]-pi_conbody),
        y[2],
        a*(a/4*(R_body-y[1])-y[2])]
    return dydt

def oscillator_body_08(y,t):
    dydt = [2*np.pi*v_body + r_limb_19 * W_limbtobody*np.sin(theta_limb_19-y[0]-pi_limbtobody) + r_body_07 * W_downbody*np.sin(theta_body_07-y[0]-pi_downbody) + r_body_16 * W_conbody*np.sin(theta_body_16-y[0]-pi_conbody),
        y[2],
        a*(a/4*(R_body-y[1])-y[2])]
    return dydt

def oscillator_body_09(y,t):
    dydt = [2*np.pi*v_body + r_limb_18 * W_limbtobody*np.sin(theta_limb_18-y[0]-pi_limbtobody) + r_body_10 * W_upbody*np.sin(theta_body_10-y[0]-pi_upbody) + r_body_01 * W_conbody*np.sin(theta_body_01-y[0]-pi_conbody),
        y[2],
        a*(a/4*(R_body-y[1])-y[2])]
    return dydt

def oscillator_body_10(y,t):
    dydt = [2*np.pi*v_body + r_limb_18 * W_limbtobody*np.sin(theta_limb_18-y[0]-pi_limbtobody) + r_body_09 * W_downbody*np.sin(theta_body_09-y[0]-pi_downbody) + r_body_11 * W_upbody*np.sin(theta_body_11-y[0]-pi_upbody)+ r_body_02 * W_conbody*np.sin(theta_body_02-y[0]-pi_conbody),
        y[2],
        a*(a/4*(R_body-y[1])-y[2])]
    return dydt

def oscillator_body_11(y,t):
    dydt = [2*np.pi*v_body + r_limb_18 * W_limbtobody*np.sin(theta_limb_18-y[0]-pi_limbtobody) + r_body_10 * W_downbody*np.sin(theta_body_10-y[0]-pi_downbody) + r_body_12 * W_upbody*np.sin(theta_body_12-y[0]-pi_upbody)+ r_body_03 * W_conbody*np.sin(theta_body_03-y[0]-pi_conbody),
        y[2],
        a*(a/4*(R_body-y[1])-y[2])]
    return dydt

def oscillator_body_12(y,t):
    dydt = [2*np.pi*v_body + r_limb_18 * W_limbtobody*np.sin(theta_limb_18-y[0]-pi_limbtobody) + r_body_11 * W_downbody*np.sin(theta_body_11-y[0]-pi_downbody) + r_body_13 * W_upbody*np.sin(theta_body_13-y[0]-pi_upbody)+ r_body_04 * W_conbody*np.sin(theta_body_04-y[0]-pi_conbody),
        y[2],
        a*(a/4*(R_body-y[1])-y[2])]
    return dydt

def oscillator_body_13(y,t):
    dydt = [2*np.pi*v_body + r_limb_20 * W_limbtobody*np.sin(theta_limb_20-y[0]-pi_limbtobody) + r_body_12 * W_downbody*np.sin(theta_body_12-y[0]-pi_downbody) + r_body_14 * W_upbody*np.sin(theta_body_14-y[0]-pi_upbody)+ r_body_05 * W_conbody*np.sin(theta_body_05-y[0]-pi_conbody),
        y[2],
        a*(a/4*(R_body-y[1])-y[2])]
    return dydt

def oscillator_body_14(y,t):
    dydt = [2*np.pi*v_body + r_limb_20 * W_limbtobody*np.sin(theta_limb_20-y[0]-pi_limbtobody) + r_body_13 * W_downbody*np.sin(theta_body_13-y[0]-pi_downbody) + r_body_15 * W_upbody*np.sin(theta_body_15-y[0]-pi_upbody)+ r_body_06 * W_conbody*np.sin(theta_body_06-y[0]-pi_conbody),
        y[2],
        a*(a/4*(R_body-y[1])-y[2])]
    return dydt

def oscillator_body_15(y,t):
    dydt = [2*np.pi*v_body + r_limb_20 * W_limbtobody*np.sin(theta_limb_20-y[0]-pi_limbtobody) + r_body_14 * W_downbody*np.sin(theta_body_14-y[0]-pi_downbody) + r_body_16 * W_upbody*np.sin(theta_body_16-y[0]-pi_upbody)+ r_body_07 * W_conbody*np.sin(theta_body_07-y[0]-pi_conbody),
        y[2],
        a*(a/4*(R_body-y[1])-y[2])]
    return dydt

def oscillator_body_16(y,t):
    dydt = [2*np.pi*v_body + r_limb_20 * W_limbtobody*np.sin(theta_limb_20-y[0]-pi_limbtobody) + r_body_15 * W_downbody*np.sin(theta_body_15-y[0]-pi_downbody) + r_body_08 * W_conbody*np.sin(theta_body_08-y[0]-pi_conbody),
        y[2],
        a*(a/4*(R_body-y[1])-y[2])]
    return dydt

def oscillator_limb_17(y,t):
    dydt = [2*np.pi*v_limb + r_limb_18 * W_withinlimb*np.sin(theta_limb_18-theta_limb_17-pi_withinlimb)+ r_limb_19 * W_withinlimb*np.sin(theta_limb_19-theta_limb_17-pi_withinlimb),
        rdot_limb_17,
        a*(a/4*(R_limb-r_limb_17)-rdot_limb_17)]
    return dydt

def oscillator_limb_18(y,t):
    dydt = [2*np.pi*v_limb + r_limb_17 * W_withinlimb*np.sin(theta_limb_17-theta_limb_18-pi_withinlimb)+ r_limb_20 * W_withinlimb*np.sin(theta_limb_20-theta_limb_18-pi_withinlimb),
        rdot_limb_18,
        a*(a/4*(R_limb-r_limb_18)-rdot_limb_18)]
    return dydt

def oscillator_limb_19(y,t):
    dydt = [2*np.pi*v_limb + r_limb_17 * W_withinlimb*np.sin(theta_limb_17-theta_limb_19-pi_withinlimb)+ r_limb_20 * W_withinlimb*np.sin(theta_limb_20-theta_limb_19-pi_withinlimb),
        rdot_limb_19,
        a*(a/4*(R_limb-r_limb_19)-rdot_limb_19)]
    return dydt

def oscillator_limb_20(y,t):
    dydt = [2*np.pi*v_limb + r_limb_18 * W_withinlimb*np.sin(theta_limb_18-theta_limb_20-pi_withinlimb)+ r_limb_19 * W_withinlimb*np.sin(theta_limb_19-theta_limb_20-pi_withinlimb),
        rdot_limb_20,
        a*(a/4*(R_limb-r_limb_20)-rdot_limb_20)]
    return dydt

def ode():
    global theta_body_01
    global r_body_01
    global rdot_body_01
    global x_body_01
    global theta_body_02
    global r_body_02 
    global rdot_body_02
    global x_body_02
    global theta_body_03
    global r_body_03
    global rdot_body_03
    global x_body_03
    global theta_body_04 
    global r_body_04
    global rdot_body_04
    global x_body_04
    global theta_body_05
    global r_body_05
    global rdot_body_05
    global x_body_05
    global theta_body_06
    global r_body_06
    global rdot_body_06
    global x_body_06
    global theta_body_07
    global r_body_07
    global rdot_body_07
    global x_body_07
    global theta_body_08
    global r_body_08
    global rdot_body_08 
    global x_body_08
    global theta_body_09
    global r_body_09
    global rdot_body_09
    global x_body_09
    global theta_body_10 
    global r_body_10
    global rdot_body_10 
    global x_body_10
    global theta_body_11 
    global r_body_11
    global rdot_body_11
    global x_body_11
    global theta_body_12
    global r_body_12
    global rdot_body_12
    global x_body_12
    global theta_body_13
    global r_body_13
    global rdot_body_13
    global x_body_13
    global theta_body_14
    global r_body_14
    global rdot_body_14
    global x_body_14
    global theta_body_15
    global r_body_15
    global rdot_body_15
    global x_body_15
    global theta_body_16
    global r_body_16
    global rdot_body_16 
    global x_body_16
    global theta_limb_17 
    global r_limb_17
    global rdot_limb_17 
    global x_limb_17
    global theta_limb_18 
    global r_limb_18
    global rdot_limb_18
    global x_limb_18
    global theta_limb_19
    global r_limb_19
    global rdot_limb_19
    global x_limb_19
    global theta_limb_20
    global r_limb_20
    global rdot_limb_20
    global x_limb_20
    global alpha_1
    global alpha_2
    global alpha_3
    global alpha_4
    global alpha_5
    global alpha_6
    global alpha_7
    global alpha_8
    global d_walk 
    global d_swim 
    global c_v1_body  
    global c_v0_body  
    global c_v1_limb_w 
    global c_v1_limb_s 
    global c_v0_limb_w 
    global c_v0_limb_s 
    global c_R1_body  
    global c_R0_body  
    global c_R1_limb_w 
    global c_R0_limb_w 
    global c_R1_limb_s 
    global c_R0_limb_s 
    global a
    global W_downbody
    global pi_downbody
    global W_upbody
    global pi_upbody
    global W_conbody
    global pi_conbody
    global W_limbtobody
    global pi_limbtobody
    global W_withinlimb
    global pi_withinlimb
    global v_body
    global v_limb
    global R_body
    global R_limb

    sol_01 = odeint(oscillator_body_01,[theta_body_01, r_body_01, rdot_body_01], [0, 0.02])
    theta_body_01 = sol_01[1,0]
    r_body_01 = sol_01[1,1]
    rdot_body_01 = sol_01[1,2]
    x_body_01 =r_body_01 * (1+np.cos(theta_body_01))
    
    sol_02 = odeint(oscillator_body_02, [theta_body_02, r_body_02, rdot_body_02], [0, 0.02])
    theta_body_02 = sol_02[1,0]
    r_body_02 = sol_02[1,1]
    rdot_body_02 = sol_02[1,2]
    x_body_02 =r_body_02 * (1+np.cos(theta_body_02))
    
    sol_03 = odeint(oscillator_body_03,[theta_body_03, r_body_03, rdot_body_03], [0, 0.02])
    theta_body_03 = sol_03[1,0]
    r_body_03 = sol_03[1,1]
    rdot_body_03 = sol_03[1,2]
    x_body_03 =r_body_03 * (1+np.cos(theta_body_03))
    
    sol_04 = odeint(oscillator_body_04, [theta_body_04, r_body_04, rdot_body_04], [0, 0.02])
    theta_body_04 = sol_04[1,0]
    r_body_04 = sol_04[1,1]
    rdot_body_04 = sol_04[1,2]
    x_body_04 =r_body_04 * (1+np.cos(theta_body_04))
    
    sol_05 = odeint(oscillator_body_05, [theta_body_05, r_body_05, rdot_body_05], [0, 0.02])
    theta_body_05 = sol_05[1,0]
    r_body_05 = sol_05[1,1]
    rdot_body_05 = sol_05[1,2]
    x_body_05 =r_body_05 * (1+np.cos(theta_body_05))
    
    sol_06 = odeint(oscillator_body_06, [theta_body_06, r_body_06, rdot_body_06], [0, 0.02])
    theta_body_06 = sol_06[1,0]
    r_body_06 = sol_06[1,1]
    rdot_body_06 = sol_06[1,2]
    x_body_06 =r_body_06 * (1+np.cos(theta_body_06))
    
    sol_07 = odeint(oscillator_body_07, [theta_body_07, r_body_07, rdot_body_07], [0, 0.02])
    theta_body_07 = sol_07[1,0]
    r_body_07 = sol_07[1,1]
    rdot_body_07 = sol_07[1,2]
    x_body_07 =r_body_07 * (1+np.cos(theta_body_07))
    
    sol_08 = odeint(oscillator_body_08, [theta_body_08, r_body_08, rdot_body_08], [0, 0.02])
    theta_body_08 = sol_08[1,0]
    r_body_08 = sol_08[1,1]
    rdot_body_08 = sol_08[1,2]
    x_body_08 =r_body_08 * (1+np.cos(theta_body_08))
    
    sol_09 = odeint(oscillator_body_09, [theta_body_09, r_body_09, rdot_body_09], [0, 0.02])
    theta_body_09 = sol_09[1,0]
    r_body_09 = sol_09[1,1]
    rdot_body_09 = sol_09[1,2]
    x_body_09 =r_body_09 * (1+np.cos(theta_body_09))
    
    sol_10 = odeint(oscillator_body_10, [theta_body_10, r_body_10, rdot_body_10], [0, 0.02])
    theta_body_10 = sol_10[1,0]
    r_body_10 = sol_10[1,1]
    rdot_body_10 = sol_10[1,2]
    x_body_10 =r_body_10 * (1+np.cos(theta_body_10))
    
    sol_11 = odeint(oscillator_body_11, [theta_body_11, r_body_11, rdot_body_11], [0, 0.02])
    theta_body_11 = sol_11[1,0]
    r_body_11 = sol_11[1,1]
    rdot_body_11 = sol_11[1,2]
    x_body_11 =r_body_11 * (1+np.cos(theta_body_11))
    
    sol_12 = odeint(oscillator_body_12, [theta_body_12, r_body_12, rdot_body_12], [0, 0.02])
    theta_body_12 = sol_12[1,0]
    r_body_12 = sol_12[1,1]
    rdot_body_12 = sol_12[1,2]
    x_body_12 =r_body_12 * (1+np.cos(theta_body_12))
    
    sol_13 = odeint(oscillator_body_13, [theta_body_13, r_body_13, rdot_body_13], [0, 0.02])
    theta_body_13 = sol_13[1,0]
    r_body_13 = sol_13[1,1]
    rdot_body_13 = sol_13[1,2]
    x_body_13 =r_body_13 * (1+np.cos(theta_body_13))
    
    sol_14 = odeint(oscillator_body_14, [theta_body_14, r_body_14, rdot_body_14], [0, 0.02])
    theta_body_14 = sol_14[1,0]
    r_body_14 = sol_14[1,1]
    rdot_body_14 = sol_14[1,2]
    x_body_14 =r_body_14 * (1+np.cos(theta_body_14))
    
    sol_15 = odeint(oscillator_body_15, [theta_body_15, r_body_15, rdot_body_15], [0, 0.02])
    theta_body_15 = sol_15[1,0]
    r_body_15 = sol_15[1,1]
    rdot_body_15 = sol_15[1,2]
    x_body_15 =r_body_15 * (1+np.cos(theta_body_15))
    
    sol_16 = odeint(oscillator_body_16, [theta_body_16, r_body_16, rdot_body_16], [0, 0.02])
    theta_body_16 = sol_16[1,0]
    r_body_16 = sol_16[1,1]
    rdot_body_16 = sol_16[1,2]
    x_body_16 =r_body_16 * (1+np.cos(theta_body_16))
    
    sol_17 = odeint(oscillator_limb_17, [theta_limb_17, r_limb_17, rdot_limb_17], [0, 0.02])
    theta_limb_17 = sol_17[1,0]
    r_limb_17 = sol_17[1,1]
    rdot_limb_17 = sol_17[1,2]
    x_limb_17 =r_limb_17 * (1+np.cos(theta_limb_17))
    
    sol_18 = odeint(oscillator_limb_18, [theta_limb_18, r_limb_18, rdot_limb_18], [0, 0.02])
    theta_limb_18 = sol_18[1,0]
    r_limb_18 = sol_18[1,1]
    rdot_limb_18 = sol_18[1,2]
    x_limb_18 =r_limb_18 * (1+np.cos(theta_limb_18))
    
    sol_19 = odeint(oscillator_limb_19, [theta_limb_19, r_limb_19, rdot_limb_19], [0, 0.02])
    theta_limb_19 = sol_19[1,0]
    r_limb_19 = sol_19[1,1]
    rdot_limb_19 = sol_19[1,2]
    x_limb_19 =r_limb_19 * (1+np.cos(theta_limb_19))
    
    sol_20 = odeint(oscillator_limb_20, [theta_limb_20, r_limb_20, rdot_limb_20], [0, 0.02])
    theta_limb_20 = sol_20[1,0]
    r_limb_20 = sol_20[1,1]
    rdot_limb_20 = sol_20[1,2]
    x_limb_20 =r_limb_20 * (1+np.cos(theta_limb_20))

    pi_1 = alpha_1*(x_body_02 - x_body_10)
    pi_2 = alpha_2*(x_body_03 - x_body_11)
    pi_3 = alpha_3*(x_body_04 - x_body_12)
    pi_4 = alpha_4*(x_body_06 - x_body_14)
    pi_5 = alpha_5*(x_body_07 - x_body_15)
    pi_6 = alpha_6*(x_body_08 - x_body_16)    
    pi_7 = theta_limb_18
    pi_8 = theta_limb_17
    pi_9 = theta_limb_20
    pi_10 = theta_limb_19
    return [pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8, pi_9, pi_10]

def OU(x, mu, theta, sigma):
    return theta *(mu - x) + sigma * np.random.randn(1)

def step(terminate_step, que_pos, que_avg, que_rwd, actor):
    global theta_body_01
    global r_body_01
    global rdot_body_01
    global x_body_01
    global theta_body_02
    global r_body_02 
    global rdot_body_02
    global x_body_02
    global theta_body_03
    global r_body_03
    global rdot_body_03
    global x_body_03
    global theta_body_04 
    global r_body_04
    global rdot_body_04
    global x_body_04
    global theta_body_05
    global r_body_05
    global rdot_body_05
    global x_body_05
    global theta_body_06
    global r_body_06
    global rdot_body_06
    global x_body_06
    global theta_body_07
    global r_body_07
    global rdot_body_07
    global x_body_07
    global theta_body_08
    global r_body_08
    global rdot_body_08 
    global x_body_08
    global theta_body_09
    global r_body_09
    global rdot_body_09
    global x_body_09
    global theta_body_10 
    global r_body_10
    global rdot_body_10 
    global x_body_10
    global theta_body_11 
    global r_body_11
    global rdot_body_11
    global x_body_11
    global theta_body_12
    global r_body_12
    global rdot_body_12
    global x_body_12
    global theta_body_13
    global r_body_13
    global rdot_body_13
    global x_body_13
    global theta_body_14
    global r_body_14
    global rdot_body_14
    global x_body_14
    global theta_body_15
    global r_body_15
    global rdot_body_15
    global x_body_15
    global theta_body_16
    global r_body_16
    global rdot_body_16 
    global x_body_16
    global theta_limb_17 
    global r_limb_17
    global rdot_limb_17 
    global x_limb_17
    global theta_limb_18 
    global r_limb_18
    global rdot_limb_18
    global x_limb_18
    global theta_limb_19
    global r_limb_19
    global rdot_limb_19
    global x_limb_19
    global theta_limb_20
    global r_limb_20
    global rdot_limb_20
    global x_limb_20
    global alpha_1
    global alpha_2
    global alpha_3
    global alpha_4
    global alpha_5
    global alpha_6
    global alpha_7
    global alpha_8
    global d_walk 
    global d_swim 
    global c_v1_body  
    global c_v0_body  
    global c_v1_limb_w 
    global c_v1_limb_s 
    global c_v0_limb_w 
    global c_v0_limb_s 
    global c_R1_body  
    global c_R0_body  
    global c_R1_limb_w 
    global c_R0_limb_w 
    global c_R1_limb_s 
    global c_R0_limb_s 
    global a
    global W_downbody
    global pi_downbody
    global W_upbody
    global pi_upbody
    global W_conbody
    global pi_conbody
    global W_limbtobody
    global pi_limbtobody
    global W_withinlimb
    global pi_withinlimb
    global v_body
    global v_limb
    global R_body
    global R_limb
    global score
    global done_step
    global epsilon

    s2 = np.array([0.0 for _ in range(7)])
    flag = False
    env = np.array([0.0 for _ in range(3)])
    que_env = queue(50,3,False)
    
    reward = 0.0
    learning_done = False
    done = False
    terminate = False
    i=0
    
    while True:
        rate.sleep()
        epsilon -= 1.0/ EXPLORE
        action_noise = np.zeros([1, actor.a_dim])
        
        i = i+1
        centroid = [(0.2*pose_x1 + 0.1*pose_x5)/0.3 , (0.2*pose_y1 + 0.1*pose_y5)/ 0.3]
        vel_x = (vel_x1 + vel_x5)/2.0
        vel_y = (vel_y1 + vel_y5)/2.0

        if centroid[1] < 1.25 :
            environment = env_ground
        elif centroid[1] >= 1.25 and centroid[1] <3.15:
            environment = env_mud
        elif centroid[1] >=3.15 and centroid[1] <5.35:
            environment = env_water
        else:
            environment = env_ground

        
        s_cur = np.array([[environment[0][0],environment[0][1], environment[0][2], centroid[0], centroid[1], vel_x, vel_y]])
        #action
        action = actor.predict(s_cur)
        action_noise[0][6] = max(epsilon, 0) * OU(action[0][6], 0.2, 0.1, 0.05)
        action_noise[0][7] = max(epsilon, 0) * OU(action[0][7], 0.0 , 0.1, 0.0001)
        action_noise[0][8] = max(epsilon, 0) * OU(action[0][8], 0.0 , 0.1, 0.0001)
        action_noise[0][13] = max(epsilon, 0) * OU(action[0][13], 0.0 , 0.1, 0.0001)
        action_noise[0][14] = max(epsilon, 0) * OU(action[0][14], 0.0 , 0.1, 0.0001)
        action = action + action_noise
        #action[0][0], action[0][1],action[0][2],action[0][3],action[0][4],action[0][5],action[0][6],action[0][7],action[0][8],action[0][9],action[0][10],action[0][11],action[0][12],action[0][13],action[0][14],action[0][15],action[0][16],action[0][17],action[0][18],action[0][19],action[0][20],action[0][21],action[0][22],action[0][23],action[0][24],action[0][25],action[0][26],action[0][27]
        alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, d, c_v1_body,c_v0_body, c_v1_limb, c_v0_limb, c_R1_body, c_R0_body, c_R1_limb, c_R0_limb, a, W_downbody,  W_upbody, W_conbody, W_limbtobody,  W_withinlimb, pi_downbody,pi_upbody, pi_conbody, pi_limbtobody,pi_withinlimb = actor.scaleshift( action[0] )
        
        if centroid[1] >= 1.25 and centroid[1] <3.15:
            drive_sig.append(d)
        
        v_body = c_v1_body * d + c_v0_body
        v_limb = c_v1_limb * d + c_v0_limb
        R_body = c_R1_body * d + c_R0_body
        R_limb = c_R1_limb * d + c_R0_limb
        
        pi = ode()
        
        pub_msg_body1 = Float64(data=pi[0])
        pub_msg_body2 = Float64(data=pi[1])
        pub_msg_body3 = Float64(data=pi[2])
        pub_msg_body4 = Float64(data=pi[3])
        pub_msg_body5 = Float64(data=pi[4])
        pub_msg_body6 = Float64(data=pi[5])
        pub_msg_limb1 = Float64(data=pi[6])
        pub_msg_limb2 = Float64(data=pi[7])
        pub_msg_limb3 = Float64(data=pi[8])
        pub_msg_limb4 = Float64(data=pi[9])

        pub_body1.publish(pub_msg_body1)
        pub_body2.publish(pub_msg_body2)   
        pub_body3.publish(pub_msg_body3)
        pub_body4.publish(pub_msg_body4)
        pub_body5.publish(pub_msg_body5)
        pub_body6.publish(pub_msg_body6)
        pub_limb1.publish(pub_msg_limb1)
        pub_limb2.publish(pub_msg_limb2)
        pub_limb3.publish(pub_msg_limb3)
        pub_limb4.publish(pub_msg_limb4)          
        
        # get s2
        position = np.array([centroid[0], centroid[1], vel_x , vel_y])
        pos_flag = que_pos.push( position )
        que_env.push( environment )
        
        if pos_flag == True :
            avg_flag = que_avg.push( que_pos.avr() )
        
        avg_data = que_avg.data()

        if pos_flag and avg_flag == True:
            print(alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, d, c_v1_body,c_v0_body, c_v1_limb, c_v0_limb, c_R1_body, c_R0_body, c_R1_limb, c_R0_limb, a, W_downbody,  W_upbody, W_conbody, W_limbtobody,  W_withinlimb, pi_downbody,pi_upbody, pi_conbody, pi_limbtobody,pi_withinlimb)
            #print("v_limb: {:f}, R_limb: {:f}".format(v_limb, R_body) )
            env = que_env.avr()
            s2 = np.array([[env[0],env[1], env[2], avg_data[0][0], avg_data[0][1], avg_data[0][2], avg_data[0][3] ]])
            reward = avg_data[0][3] - np.abs(avg_data[0][2])
            
            
            if abs(centroid[0]) > 1.0 or centroid[1] < -1.0:
                print("Off the track")
                reward = -10.0
                terminate = True
            elif terminate_step > 25  :
                print("Time's up")
                reward = -15.0
                terminate = True

            elif abs(centroid[0]) < 1.0 and centroid[1] > 3.10:
                print("Goal!")
                reward = 10.0
                done_step += 1
                print("done_step: {:d}".format(done_step))
                terminate = True
                learning_done = True

            break
        
    return action, s2, reward, terminate, learning_done, done_step,{}


def reset():
    # rosservice reset world
    reset_jointstate('rrbot', 'rrbot_description', ['rrbot__body_joint_1','rrbot__body_joint_2','rrbot__body_joint_3','rrbot__body_joint_4','rrbot__body_joint_5','rrbot__body_joint_6','rrbot__limb_joint_1','rrbot__limb_joint_2','rrbot__limb_joint_3','rrbot__limb_joint_4'], [0, 0, 0, 0, 0, 0, 3.14, 0 ,0 , 3.14])
    #,'rrbot__limb_joint_1','rrbot__limb_joint_2','rrbot__limb_joint_3','rrbot__limb_joint_4' , , 0, 0, 0, 0
    
    time.sleep(0.5)
    reset_world()
    rate.sleep()
    
    # get state
    environment = env_ground
    centroid = [(0.2*pose_x1 + 0.1*pose_x5)/0.3 , (0.2*pose_y1 + 0.1*pose_y5)/ 0.3]
    vel_x = (vel_x1 + vel_x5)/2.0
    vel_y = (vel_y1 + vel_y5)/2.0
    s = np.array([[environment[0][0], environment[0][1], environment[0][2], centroid[0], centroid[1], vel_x, vel_y]])
    return s


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax_Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


def train(sess, graph, args, actor, critic):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()
    
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)  
    

    sess.run(tf.global_variables_initializer())
    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
    que_pos = queue(50,4,True)
    que_avg = queue(1,4,False)
    que_rwd = queue(20,1,False)
    
    for i in range(101):
        #, summary_pre 
        loss = actor.pretrain(x,y)
        print("pre-train step : {:d}    loss : {:f}".format(i+1, loss))
        # if i % 10 == 0:
        #     actor.writer_pre.add_summary(summary_pre, i)
        #     actor.writer_pre.flush()
    
    #pretrain result = > tensorboard
    
    for i in range(int(args['max_episodes'])):

        s = reset()
        ep_reward = 0
        step_reward = 0
        ep_ave_max_q = 0
        m =0

        while not rospy.is_shutdown():
            
            print(m)
            action, s2, r, terminal, learning_done, done_step, info = step(m, que_pos, que_avg, que_rwd ,actor)
            m += 1
            print(s2, r)
            t_.append(m)
            #print(action)
            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(action, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            step_reward += r
            print(step_reward)
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))
                
                print(np.amax(predicted_q_value))
                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()
            
            s = s2
            
            ep_reward += step_reward
            
            
            if learning_done and done_step > 30:
                plt.figure(1)
                plt.plot(t_, drive_signal)
                plt.xlabel('time(s)')
                plt.ylabel('drive signal')
                plt.show()

                return


            if terminal:
                summary_str = sess.run(summary_ops, feed_dict={summary_vars[0]: ep_reward, summary_vars[1]: ep_ave_max_q / float(m)})
                writer.add_summary(summary_str, i)
                writer.flush()
                # que_rwd.clear()
                # que_pos.clear()
                # que_avg.clear()
                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), i, (ep_ave_max_q / float(m))))
                break
            
            



def main(args):

    graph = tf.Graph()

    with graph.as_default():

        sess = tf.Session()

        with sess.as_default():
        
            np.random.seed(int(args['random_seed']))
            tf.set_random_seed(int(args['random_seed']))
            
            state_dim = 7
            action_dim = 26
            
            actor = ActorNetwork(sess, graph, state_dim, action_dim,
                                float(args['actor_lr']), float(args['tau']),
                                int(args['minibatch_size']))

            critic = CriticNetwork(sess, state_dim, action_dim,
                                float(args['critic_lr']), float(args['tau']),
                                float(args['gamma']),
                                actor.get_num_trainable_vars())

            train(sess, graph, args, actor, critic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=5000)
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='salamander_result')
    parser.add_argument('--log-dir', help='directory for storing tensorboard info', default='tf_log')

    args = vars(parser.parse_args())
    
    pp.pprint(args)

main(args)


            # action_noise[0][0] = max(epsilon, 0) * OU(a[0][0], 0.5, 0.6, 0.01)
            # action_noise[0][1] = max(epsilon, 0) * OU(a[0][1], 0.6, 0.6, 0.01)
            # action_noise[0][2] = max(epsilon, 0) * OU(a[0][2], 0.7, 0.6, 0.01)
            # action_noise[0][3] = max(epsilon, 0) * OU(a[0][3], 0.8, 0.6, 0.01)
            # action_noise[0][4] = max(epsilon, 0) * OU(a[0][4], 0.9, 0.6, 0.01)
            # action_noise[0][5] = max(epsilon, 0) * OU(a[0][5], 1.0, 0.6, 0.01)
            # action_noise[0][6] = max(epsilon, 0) * OU(a[0][6], 0.5, 0.6, 0.01)
            # action_noise[0][7] = max(epsilon, 0) * OU(a[0][7], 0.7, 0.6, 0.01)
            # action_noise[0][8] = max(epsilon, 0) * OU(a[0][8], 4.0, 0.1, 0.01)
            # action_noise[0][9] = max(epsilon, 0) * OU(a[0][9], 0.2, 0.60, 0.01)
            # action_noise[0][10] = max(epsilon, 0) * OU(a[0][10], 0.3, 0.60, 0.01)
            # action_noise[0][11] = max(epsilon, 0) * OU(a[0][11], 0.2, 0.60, 0.01)
            # action_noise[0][12] = max(epsilon, 0) * OU(a[0][12], 0.0, 0.60, 0.01)
            # action_noise[0][13] = max(epsilon, 0) * OU(a[0][13], 0.065, 0.60, 0.01)
            # action_noise[0][14] = max(epsilon, 0) * OU(a[0][14], 0.196, 0.60, 0.01)
            # action_noise[0][15] = max(epsilon, 0) * OU(a[0][15], 0.131, 0.60, 0.01)
            # action_noise[0][16] = max(epsilon, 0) * OU(a[0][16], 0.131, 0.60, 0.01)
            # action_noise[0][17] = max(epsilon, 0) * OU(a[0][17], 20, 0.1, 0.01)
            # action_noise[0][18] = max(epsilon, 0) * OU(a[0][18], 10, 0.1, 0.01)
            # action_noise[0][19] = max(epsilon, 0) * OU(a[0][19], 10, 0.1, 0.01)
            # action_noise[0][20] = max(epsilon, 0) * OU(a[0][20], 10, 0.1, 0.01)
            # action_noise[0][21] = max(epsilon, 0) * OU(a[0][21], 30, 0.1, 0.01)
            # action_noise[0][22] = max(epsilon, 0) * OU(a[0][22], 10, 0.1, 0.01)
            # action_noise[0][23] = max(epsilon, 0) * OU(a[0][23], -2*np.pi/8, 0.5, 0.01)
            # action_noise[0][24] = max(epsilon, 0) * OU(a[0][24], 2*np.pi/8, 0.5, 0.01)
            # action_noise[0][25] = max(epsilon, 0) * OU(a[0][25], np.pi, 0.5, 0.01)
            # action_noise[0][26] = max(epsilon, 0) * OU(a[0][26], np.pi, 0.5, 0.01)
            # action_noise[0][27] = max(epsilon, 0) * OU(a[0][27], np.pi, 0.5, 0.01)
            
            # a_t[0][0] = a[0][0] + action_noise[0][0]
            # a_t[0][1] = a[0][1] + action_noise[0][1]
            # a_t[0][2] = a[0][2] + action_noise[0][2]
            # a_t[0][3] = a[0][3] + action_noise[0][3]
            # a_t[0][4] = a[0][4] + action_noise[0][4]
            # a_t[0][5] = a[0][5] + action_noise[0][5]
            # a_t[0][6] = a[0][6] + action_noise[0][6]
            # a_t[0][7] = a[0][7] + action_noise[0][7]
            # a_t[0][8] = a[0][8] + action_noise[0][8]
            # a_t[0][9] = a[0][9] + action_noise[0][9]
            # a_t[0][10] = a[0][10] + action_noise[0][10]
            # a_t[0][11] = a[0][11] + action_noise[0][11]
            # a_t[0][12] = a[0][12] + action_noise[0][12]
            # a_t[0][13] = a[0][13] + action_noise[0][13]
            # a_t[0][14] = a[0][14] + action_noise[0][14]
            # a_t[0][15] = a[0][15] + action_noise[0][15]
            # a_t[0][16] = a[0][16] + action_noise[0][16]
            # a_t[0][17] = a[0][17] + action_noise[0][17]
            # a_t[0][18] = a[0][18] + action_noise[0][18]
            # a_t[0][19] = a[0][19] + action_noise[0][19]
            # a_t[0][20] = a[0][20] + action_noise[0][20]
            # a_t[0][21] = a[0][21] + action_noise[0][21]
            # a_t[0][22] = a[0][22] + action_noise[0][22]
            # a_t[0][23] = a[0][23] + action_noise[0][23]
            # a_t[0][24] = a[0][24] + action_noise[0][24]
            # a_t[0][25] = a[0][25] + action_noise[0][25]
            # a_t[0][26] = a[0][26] + action_noise[0][26]
            # a_t[0][27] = a[0][27] + action_noise[0][27]

