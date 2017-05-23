from __future__ import print_function

import os
import re
import sys
import cv2
import math
import time
from torch import np
import urllib
import util
import torch

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from config_reader import config_reader
from scipy.ndimage.filters import maximum_filter

weight_name = 'model/pose_model.pth'

blocks = {}

# find connection in the specified sequence, center 29 is in the position 15
# defines indexes (1 indexed) into output tensor of limb location predicitions
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]
           
# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
          [55,56], [37,38], [45,46]]
          
# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

bone_map = [
    'left_shoulder',  #0
    'right_shoulder', #1
    'left_upper_arm', #2
    'left_lower_arm', #3
    'right_uppder_arm', #4
    'right_lower_arm', #5
    'left_spine', #6
    'left_upper_leg', #7
    'left_lower_leg', #8
    'right_spine', #9
    'right_upper_leg', #10
    'right_lower_leg', #11
    'neck', #12
    'left_eye', #13
    'left_ear_eye', #14
    'right_eye', #15,
    'right_eye_ear' #16
]

N_JOINTS = 18

block0  = [{'conv1_1':[3,64,3,1,1]},{'conv1_2':[64,64,3,1,1]},{'pool1_stage1':[2,2,0]},{'conv2_1':[64,128,3,1,1]},{'conv2_2':[128,128,3,1,1]},{'pool2_stage1':[2,2,0]},{'conv3_1':[128,256,3,1,1]},{'conv3_2':[256,256,3,1,1]},{'conv3_3':[256,256,3,1,1]},{'conv3_4':[256,256,3,1,1]},{'pool3_stage1':[2,2,0]},{'conv4_1':[256,512,3,1,1]},{'conv4_2':[512,512,3,1,1]},{'conv4_3_CPM':[512,256,3,1,1]},{'conv4_4_CPM':[256,128,3,1,1]}]

blocks['block1_1']  = [{'conv5_1_CPM_L1':[128,128,3,1,1]},{'conv5_2_CPM_L1':[128,128,3,1,1]},{'conv5_3_CPM_L1':[128,128,3,1,1]},{'conv5_4_CPM_L1':[128,512,1,1,0]},{'conv5_5_CPM_L1':[512,38,1,1,0]}]

blocks['block1_2']  = [{'conv5_1_CPM_L2':[128,128,3,1,1]},{'conv5_2_CPM_L2':[128,128,3,1,1]},{'conv5_3_CPM_L2':[128,128,3,1,1]},{'conv5_4_CPM_L2':[128,512,1,1,0]},{'conv5_5_CPM_L2':[512,19,1,1,0]}]

for i in range(2,7):
    blocks['block%d_1'%i]  = [{'Mconv1_stage%d_L1'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L1'%i:[128,128,7,1,3]},
{'Mconv5_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L1'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L1'%i:[128,38,1,1,0]}]
    blocks['block%d_2'%i]  = [{'Mconv1_stage%d_L2'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L2'%i:[128,128,7,1,3]},
{'Mconv5_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L2'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L2'%i:[128,19,1,1,0]}]

def make_layers(cfg_dict):
    layers = []
    for i in range(len(cfg_dict)-1):
        one_ = cfg_dict[i]
        for k,v in one_.iteritems():      
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    one_ = cfg_dict[-1].keys()
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)
    
layers = []
for i in range(len(block0)):
    one_ = block0[i]
    for k,v in one_.iteritems():      
        if 'pool' in k:
            layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
            layers += [conv2d, nn.ReLU(inplace=True)]  
       
models = {}           
models['block0']=nn.Sequential(*layers)        

for k,v in blocks.iteritems():
    models[k] = make_layers(v)
                
class pose_model(nn.Module):
    def __init__(self,model_dict,transform_input=False):
        super(pose_model, self).__init__()
        self.model0   = model_dict['block0']
        self.model1_1 = model_dict['block1_1']        
        self.model2_1 = model_dict['block2_1']  
        self.model3_1 = model_dict['block3_1']  
        self.model4_1 = model_dict['block4_1']  
        self.model5_1 = model_dict['block5_1']  
        self.model6_1 = model_dict['block6_1']  
        
        self.model1_2 = model_dict['block1_2']        
        self.model2_2 = model_dict['block2_2']  
        self.model3_2 = model_dict['block3_2']  
        self.model4_2 = model_dict['block4_2']  
        self.model5_2 = model_dict['block5_2']  
        self.model6_2 = model_dict['block6_2']
        
    def forward(self, x):    
        out1 = self.model0(x)
        
        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2  = torch.cat([out1_1,out1_2,out1],1)
        
        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3   = torch.cat([out2_1,out2_2,out1],1)
        
        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4   = torch.cat([out3_1,out3_2,out1],1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5   = torch.cat([out4_1,out4_2,out1],1)  
        
        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6   = torch.cat([out5_1,out5_2,out1],1)         
              
        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)
        
        return out6_1,out6_2        


model = pose_model(models)     
script_dir = os.path.dirname(os.path.realpath(__file__))
model.load_state_dict(torch.load(os.path.join(script_dir, weight_name)))
model.cuda()
model.eval()
model.requires_grad = False

param_, model_ = config_reader()

def max_erode(x_t):
    return x_t.float() * torch.eq(x_t.float(), F.max_pool2d(x_t.float(), 3, 1, padding=1)).float()

def handle_one(oriImg, dont_draw=False, dont_resize=False):
    
    oriImg = np.squeeze(oriImg)
    oriImg = np.copy(oriImg)
    canvas = np.copy(oriImg)

    scale = model_['boxsize'] / float(oriImg.shape[0])
    h = int(oriImg.shape[0]*scale)
    w = int(oriImg.shape[1]*scale)


    pad_h = 0 if (h%model_['stride']==0) else model_['stride'] - (h % model_['stride']) 
    pad_w = 0 if (w%model_['stride']==0) else model_['stride'] - (w % model_['stride'])
    new_h = h+pad_h
    new_w = w+pad_w

    if dont_resize:
        imageToTest = oriImg
    else:
        imageToTest = cv2.resize(oriImg, (int(scale * oriImg.shape[0]), int(scale * oriImg.shape[1])), interpolation=cv2.INTER_CUBIC)

    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_['stride'], model_['padValue'])

    imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/255.0 - 0.5

    feed = Variable(T.from_numpy(imageToTest_padded), volatile=True).cuda()      

    p = time.time()
    output1, output2 = model(feed)

    heatmap = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])).cuda()(output2)

    paf = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])).cuda()(output1)       
    
    pool_t = time.time()
    map_ori = heatmap[0][:N_JOINTS]

    map_max = max_erode(F.avg_pool2d(map_ori, 3, stride=1, padding=1)) # torch.eq(blurred_avg_tensor, max_tensor).float() * blurred_avg_tensor
    
    peaks_binary = map_max > param_['thre1']
    all_peak_idxs = torch.nonzero(peaks_binary.data)

    if all_peak_idxs.size() == ():
        return None
    
    all_peak_idxs = all_peak_idxs.cpu().numpy()

    
    nonzero_vals = map_ori[peaks_binary].data.cpu().numpy()

    paf_avg = paf[0].data.cpu().numpy()

    return post_process(oriImg, canvas, paf_avg, all_peak_idxs, nonzero_vals, dont_draw)

def get_third_item(x):
    return x[2]

def post_process(oriImg, canvas, paf_avg, all_peak_idxs, nonzero_vals, dont_draw):

    all_peaks = [[] for x in range(N_JOINTS)]

    for i in xrange(all_peak_idxs.shape[0]):
        all_peaks[all_peak_idxs[i, 0]].append((all_peak_idxs[i, 2], all_peak_idxs[i, 1], nonzero_vals[i], i))

    PEAKT = time.time()

    connection_all = []
    special_k = []
    mid_num = 10

    KTTT = time.time()

    # don't really know how this works
    for k in xrange(len(mapIdx)):
        idxs = []
        for x in mapIdx[k]:
            idxs.append(x - 19)

        score_mid = paf_avg[idxs,:,:]
        candA = all_peaks[limbSeq[k][0]-1]
        candB = all_peaks[limbSeq[k][1]-1]
        nA = len(candA)
        nB = len(candB)

        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)
                    
                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num))
                    
                    vec_x = np.zeros(len(startend))
                    vec_y = np.zeros(len(startend))

                    for I in xrange(len(startend)):
                        vec_x[I] = score_mid[0, int(round(startend[I][1])), int(round(startend[I][0]))]
                        vec_y[I] = score_mid[1, int(round(startend[I][1])), int(round(startend[I][0]))] 

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = np.sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                    criterion1 = len(np.nonzero(score_midpts > param_['thre2'])[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=get_third_item, reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    cand_tmp = []
    for sublist in all_peaks:
        for item in sublist:
            cand_tmp.append(item)

    candidate = np.asarray(cand_tmp)

    # don't really know how this works either
    for k in xrange(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1
                
                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    #print "found = 2"
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = np.sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    #print subset.shape
    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    N_BONES = N_JOINTS - 1

    if dont_draw:
        res = {
            'found_ppl': []
        }
        for n in range(len(subset)):
            skel = {}
            for i in range(N_BONES):
                index = subset[n][np.array(limbSeq[i])-1]
                if -1 in index:
                    continue
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                bone_name = bone_map[i]
                skel[bone_name] = (
                    (X[0], Y[0]),
                    (X[1], Y[1])
                )
            res['found_ppl'].append(skel)
        return res

    p = time.time()
    canvas = canvas.copy()


    for i in range(N_JOINTS):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    stickwidth = 4

    N_BONES = N_JOINTS - 1

    for i in range(N_BONES):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])-1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            cv2.putText(canvas, str(i), (int(mY), int(mX)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 3)
            cv2.putText(canvas, str(i), (int(mY), int(mX)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    print('drawing took: ', time.time() - p)
    return canvas
