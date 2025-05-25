#!/usr/local/bin/python3
#
# Authors: [PLEASE PUT YOUR NAMES AND USER IDS HERE]
# Sripad Joshi (joshisri)
# Ice layer finder
# Based on skeleton code by D. Crandall, November 2021
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters

import sys
import imageio
import math

# calculate "Edge strength map" of an image                                                                                                                                      
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_boundary(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

def draw_asterisk(image, pt, color, thickness):
    for (x, y) in [ (pt[0]+dx, pt[1]+dy) for dx in range(-3, 4) for dy in range(-2, 3) if dx == 0 or dy == 0 or abs(dx) == abs(dy) ]:
        if 0 <= x < image.size[0] and 0 <= y < image.size[1]:
            image.putpixel((x, y), color)
    return image


# Save an image that superimposes three lines (simple, hmm, feedback) in three different colors 
# (yellow, blue, red) to the filename
def write_output_image(filename, image, simple, hmm, feedback, feedback_pt):
    new_image = draw_boundary(image, simple, (255, 255, 0), 2)
    new_image = draw_boundary(new_image, hmm, (0, 0, 255), 2)
    new_image = draw_boundary(new_image, feedback, (255, 0, 0), 2)
    new_image = draw_asterisk(new_image, feedback_pt, (255, 0, 0), 2)
    imageio.imwrite(filename, new_image)


def change_transitions_based_on_human_feedback(em,row):
    #Method to chnage the transition probabilites based on human input, didn't work well
    range_of_numbers = arange(0,len(em))
    max_of_range = max(range_of_numbers)
    result = [max_of_range-abs(x-row) for x in range_of_numbers]
    result = [0.00001 if x == 0 else x for x in result]
    quotient = 1/sum(result)
    #normal
    # changed_transistions = [0.00001 if x==0 else x*quotient for x in result]
    #log
    changed_transistions = [math.log(0.00001) if x==0 else math.log(x*quotient) for x in result]
    return changed_transistions




def compute_transition_probabilities(im):
    # range_of_numbers = arange(0,im.shape[0])
    # max_of_range = max(range_of_numbers)
    transition_probabilities = {}
    # for i in range(im.shape[0]):
    #     result = [max_of_range-abs(x-i) for x in range_of_numbers]
    #     result = [0.00001 if x == 0 else x for x in result]
    #     quotient = 1/sum(result)
    #     # normal
    #     # transition_probabilities[i] = {j:k for j,k in enumerate([(x*quotient) for x in result])}
    #     # math log 
    #     transition_probabilities[i] = {j:k for j,k in enumerate([math.log(0.0001) if x == 0 else math.log(x*quotient) for x in result])}

    # probs = [0.06,0.125,0.25,0.5,0.25,0.125,0.06]
    # probs = [0,0.05,0.2545,0.5,0.2545,0.05,0]
    probs = [0.60,0.65,0.70,0.75,0.80,0.85,0.88,0.9,0.88,0.85,0.80,0.75,0.70,0.65,0.60]
    probs = [x/sum(probs) for x in probs]


    for i in range((im.shape[0])):
        arr = zeros((im.shape[0],))
        m = 7
        j = i-1
        arr[i] = probs[m]
        
        m-=1
        while j>=0 and m>=0:
            arr[j] = probs[m]
            j-=1
            m-=1
        m = 8
        k = i+1
        while k<len(arr) and m<len(probs):
            arr[k] = probs[m]
            k+=1
            m+=1
        
        # result = [0.000000000000001 if x == 0 else x for x in arr]
        transition_probabilities[i] = {r:s for r,s in enumerate([math.log(0.0000000000001) if x == 0 else math.log(x) for x in arr])}

    return transition_probabilities


def computer_emission_probabilities(emissions):
    emission_probs = {}
    for i in range(emissions.shape[0]):
        emission_probs[i] = {k: emissions[i,k] for k in range(0,emissions.shape[1])}
    return emission_probs


def cal_probs_transmission(fro,to):
    fro = fro.astype(int)
    to = int(to)
    x = [max(fro)-abs(k-to) for k in fro]
    q = 1/sum(x)
    x = [m*q for m in x]
    x = [math.log(0.000000001) if u==0 else math.log(u) for u in x ]
    return x


def get_boundaries_using_basic_bayes(image,edge_strength,probability_to_be_used = "edges"):
    probabilities_image = []
    # print(image_array.shape[0])
    for i in range(image.shape[1]):
        column = image[:,i].astype(int)
        column_edges = edge_strength[:,i]
        mul_factor = 1/(max(column))
        mul_factor_edges = 1/max(column_edges)
        column_probabilities =array([((max(column)-x)*mul_factor) for x in column] )
        column_probabilities_edges = array([(min(column_edges)+0.00000000001 )*mul_factor_edges if x == 0 else x*mul_factor_edges for x in column_edges])#array([(x/max(column_edges)) * mul_factor_edges for x in column_edges])
        #convert to  log
        column_probabilities_edges = [math.log(x) for x in column_probabilities_edges]
        column_probabilities = [math.log(x+ 0.000000000000001) if x==0 else math.log(x) for x in column_probabilities]
        if probability_to_be_used == "edges":
            probabilities_image.append(column_probabilities_edges)
        elif probability_to_be_used == "image":
            probabilities_image.append(column_probabilities)
        elif probability_to_be_used == "both":
            probabilities_image.append([max(column_probabilities[i],column_probabilities_edges[i]) for i in range(len(column_probabilities_edges))])

    probabilities_image = array(probabilities_image) 
    probabilities_image = probabilities_image.T

    #simple
    b1 = []
    b2 = []
    for i in range(probabilities_image.shape[1]):
        sorted_list = argsort(probabilities_image[:,i])[::-1]
        s1 = sorted_list[0]
        for s in sorted_list[1:]:
            if s-s1>=10 or s-s1<=-10:
                s2 =s
                break
        
        b1.append(min(s1,s2))
        b2.append(max(s1,s2))
    return b1,b2,probabilities_image

def viterbi(image_array,probabilities_image,c):
    ## Viterbi code taken from the inclass activity solution - modified for the given problem statement
    transition = compute_transition_probabilities(image_array)
    emission = computer_emission_probabilities(probabilities_image[c:,:])    

    #states 
    states = [i for i in range(probabilities_image[c:,:].shape[0])]
    #initial
    initial = {i: math.log(1/probabilities_image.shape[0]) for i in range(probabilities_image[c:,:].shape[0])}

    #observed
    observed = [x for x in range(0,probabilities_image.shape[1])] 

    N = len(observed)
 
    V_table = {i:[0]*N for i in range(probabilities_image[c:,].shape[0])}
    which_table =  {i:[0]*N for i in range(probabilities_image[c:,].shape[0])}
    for s in states:
        #normal
        # V_table[s][0] =   initial[s] * emission[s][observed[0]] #math.log(initial[s]) + math.log(emission[s][observed[0]])#initial[s] * emission[s][observed[0]]
        #log
        V_table[s][0] =   initial[s] + emission[s][observed[0]]
   

    for i in range(1, N):
        for s in states:

           
            (which_table[s][i], V_table[s][i]) =  max( [ (s0, V_table[s0][i-1] + 
transition[s0][s]) for s0 in states ], key=lambda l:l[1] ) 


            V_table[s][i] += emission[s][observed[i]]
    viterbi_seq =[''] * N
    viterbi_seq[N-1] = argmax([ V_table[k][i] for k in range(probabilities_image[c:,].shape[0])])
    for i in range(N-2,-1,-1):
        viterbi_seq[i] =which_table[viterbi_seq[i+1]][i+1]

    return viterbi_seq


def viterbi_human_feedback(image_array,probabilities_image,c,x,y):
    # Viterbi code taken from the inclass activity solution - modified for the given problem statement
    # Idea of dividing the problem into two, and tweaking initial probabilities using human input was taken from  - 
    # https://github.com/srmanj/Artificial-Intelligence-geotagging-using-Viterbi/blob/master/mountain.py

    h_x = y-c
    if h_x <0:
        h_x = 0
    h_y = x
    if h_y == 0 or h_y>probabilities_image[c:,:].shape[1]:
        h_y = 1

    if h_x>probabilities_image[c:,:].shape[0]:
        h_x = probabilities_image[c:,:].shape[0]-1
    if h_y>probabilities_image[c:,:].shape[1]:
        h_y = probabilities_image[c:,:].shape[1]-1



    transition = compute_transition_probabilities(image_array)

    probabilities_image_a = probabilities_image[c:,:h_y]

    if h_y!=0:
        probabilities_image_a=flip(probabilities_image_a, 1)

    probabilities_image_b = probabilities_image[c:,h_y:]

    emission_a = computer_emission_probabilities(probabilities_image_a) 

    emission_b = computer_emission_probabilities(probabilities_image_b)    

    #states 
    states = [i for i in range(probabilities_image[c:,:].shape[0])]
    #initial
    init_probs = [-10]*probabilities_image[c:,:].shape[0]  

    init_probs[h_x] = 10
    initial = {i: x for i,x in enumerate(init_probs)}
    #observed
    observed_a = [x for x in range(0,h_y)]
    observed_b = [x for x in range(0,probabilities_image[c:,:].shape[1] - h_y)] 
    N_a = len(observed_a)
    N_b = len(observed_b)
    V_table_a = {i:[0]*N_a for i in range(probabilities_image[c:,].shape[0])}
    which_table_a =  {i:[0]*N_a for i in range(probabilities_image[c:,].shape[0])}

    V_table_b = {i:[0]*N_b for i in range(probabilities_image[c:,].shape[0])}
    which_table_b =  {i:[0]*N_b for i in range(probabilities_image[c:,].shape[0])}
    
    for s in states:
        #normal
        # V_table[s][0] =   initial[s] * emission[s][observed[0]] #math.log(initial[s]) + math.log(emission[s][observed[0]])#initial[s] * emission[s][observed[0]]
        #log  
        V_table_a[s][0] =   initial[s] + emission_a[s][observed_a[0]]
        V_table_b[s][0] =   initial[s] + emission_b[s][observed_b[0]]
   

    for i in range(1, N_a):
        for s in states:
            (which_table_a[s][i], V_table_a[s][i]) =  max( [ (s0, V_table_a[s0][i-1] + 
transition[s0][s]) for s0 in states ], key=lambda l:l[1] ) 
            V_table_a[s][i] += emission_a[s][observed_a[i]]
    if h_y==1:
        i = 0
    viterbi_seq_a =[''] * N_a
    viterbi_seq_a[N_a-1] = argmax([ V_table_a[k][i] for k in range(probabilities_image[c:,].shape[0])])


    for i in range(1, N_b):
        
        for s in states:
            (which_table_b[s][i], V_table_b[s][i]) =  max( [ (s0, V_table_b[s0][i-1] + 
transition[s0][s]) for s0 in states ], key=lambda l:l[1] ) 
            V_table_b[s][i] += emission_b[s][observed_b[i]]
    if N_b <2:
        i = 0
        
    viterbi_seq_b =[''] * N_b
   
    viterbi_seq_b[N_b-1] = argmax([ V_table_b[k][i] for k in range(probabilities_image[c:,].shape[0])])

    for i in range(N_a-2,-1,-1):
        viterbi_seq_a[i] =which_table_a[viterbi_seq_a[i+1]][i+1]

    for i in range(N_b-2,-1,-1):
        viterbi_seq_b[i] =which_table_b[viterbi_seq_b[i+1]][i+1]
        

    return viterbi_seq_a[::-1] + viterbi_seq_b


# main program
#
if __name__ == "__main__":

    if len(sys.argv) != 6:
        raise Exception("Program needs 5 parameters: input_file airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord")

    input_filename = sys.argv[1]
    gt_airice = [ int(i) for i in sys.argv[2:4] ]
    gt_icerock = [ int(i) for i in sys.argv[4:6] ]

    # load in image 
    input_image = Image.open(input_filename).convert('RGB')
    image_array = array(input_image.convert('L'))

    # compute edge strength mask -- in case it's helpful. Feel free to use this.
    edge_strength = edge_strength(input_image)
    imageio.imwrite('edges.png', uint8(255 * edge_strength / (amax(edge_strength))))

    # You'll need to add code here to figure out the results! For now,
    # just create some random lines.
   

    
    b1,b2,probabilities_image = get_boundaries_using_basic_bayes(image_array,edge_strength,"edges")

    #viterbi for air-ice boundary 
    air_ice_boundary = viterbi(image_array,probabilities_image.copy(),0)

     #viterbi for air-ice boundary  human feedback
    air_ice_boundary_with_feedback = viterbi_human_feedback(image_array,probabilities_image.copy(),0,gt_airice[0],gt_airice[1])

    #viterbi for ice-rock boundary 
    new_c =int( mean(air_ice_boundary)+15)

    ice_rock_boundary = viterbi(image_array,probabilities_image.copy(),new_c)

    ice_rock_boundary_feedback = viterbi_human_feedback(image_array,probabilities_image.copy(),new_c,gt_icerock[0],gt_icerock[1])
    
   
    
    
    airice_simple =b1
    airice_hmm =air_ice_boundary
    airice_feedback=air_ice_boundary_with_feedback

    icerock_simple = b2
    icerock_hmm = [new_c+x for x in ice_rock_boundary]
    icerock_feedback= [new_c+x for x in ice_rock_boundary_feedback]
    # Now write out the results as images and a text file
    write_output_image("air_ice_output.png", input_image.copy(), airice_simple, airice_hmm, airice_feedback, gt_airice)
    write_output_image("ice_rock_output.png", input_image.copy(), icerock_simple, icerock_hmm, icerock_feedback, gt_icerock)
    with open("layers_output.txt", "w") as fp:
        for i in (airice_simple, airice_hmm, airice_feedback, icerock_simple, icerock_hmm, icerock_feedback):
            fp.write(str(i) + "\n")