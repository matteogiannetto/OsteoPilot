import os
import re
import scipy.ndimage
import skimage.morphology

import cv2 as cv
import numpy as np

from glob import glob
from tqdm import tqdm
from skimage import measure
from skimage.measure import regionprops
from scipy.ndimage.measurements import find_objects


# TODO make more user friendly
path_segs = './data/T5_S33'

# Collecting image paths 
segs_path_list = glob(os.path.join(path_segs, 'predicted_seg/*.tif'))
segs_path_list.sort(key=lambda f: int(re.sub('\D', '', f)))

seg_vol = []
# Start iterating through all the images
for seg_path in tqdm(segs_path_list): 
    seg_vol.append(cv.imread(seg_path, -1))

# Create the 3D array of the segmentations
seg_vol = np.asarray(seg_vol)
seg_vol = np.transpose(seg_vol, axes=[1, 2, 0])

# Component Label
s = scipy.ndimage.morphology.generate_binary_structure(3,2) #(dimensions, connectivity)
labeled_array, num_features = scipy.ndimage.measurements.label(seg_vol,structure=s)
print (num_features)

# Get regionprops for each lacuna
Area = []
Centroid = []
Volume = []
SurfaceArea = []
MainAxisVector = np.empty((0,3))
l_x = []
l_y = []
l_z = []
#dim = ((num_features_new*3))
matrix = np.zeros((num_features,9))
Distribution_tensor_parz = np.zeros((3,3)) #create space for partial distribution tensor, after divided by (num_feauteres - 1)
Alignment_tensor_parz = np.zeros((3,3))  #create space for partial allignement tensor, after divided by (******************)
vector_Lc_Or1_x = []             #create space to collect LcOr1 referring to each lacunae
vector_Lc_Or1_y = [] 
vector_Lc_Or1_z = [] 
#LcLe=np.empty((0,3)) #create enough space to store lacune lenghts
Distance_Max = []
Distance_Min = []
Len = []
Lc_St=[] #Lacuna stretch parameter
Lc_Ob=[] #Lacuna oblateness parameter
Lc_Or_1_x = [] #lacuna orientation 1 along first axis
Lc_Or_1_y = [] #lacuna orientation 1 along second axis
Lc_Or_1_z = [] #lacuna orientation 1 along third axis 
Lc_Or_2_x = [] #lacuna orientation 2 along first axis
Lc_Or_2_y = [] #lacuna orientation 2 along second axis
Lc_Or_2_z = [] #lacuna orientation 2 along third axis
Lc_glob_St = [] #global Lacuna stretch parameter
Lc_glob_Ob = [] #global Lacuna oblateness parameter
Lc_Al = [] #lacuna alignment global parameters
#to store number of neighbors of distribution tensor
neighbors_def = [] #to store number of neighbors of distribution tensor
neighbors_ali = [] #to store number of neighbors of alignment tensor
num_neighbors_all = [] #to store neighbors of Alignment tensor
orientation_x = [] #to collect eigenvectors of neighbors
orientation_y = [] #to collect eigenvectors of neighbors
orientation_z = [] #to collect eigenvectors of neighbors
l_tot = np.zeros((num_features,3)) #to collect distance between centroids
or_tot = np.zeros((num_features,3)) #to collect eigenvectors to find alignement tensor
dim = num_features #to set the dimension
real_neighbors_final = np.zeros((dim,dim)) #to create a matrix full of zeros in which we store number of neighbors related to that specific lacuna
real_neighbors = [] #to have a space to collect temporarly lacuna


# Helper functions to avoid the coordinates going out of bounds
# and convert the coordinates to integers
# helper functions to avoid the bounding box coordinates going out of bounds
# and also to convert the coordinates to integers
def left_bound(a):
    a = int(np.around((abs(a)+a)/2))
    return a
def right_bound_x(a,scan):
    if a > scan.shape[0]:
        a = int(scan.shape[0])
    return int(np.around(a))
def right_bound_y(a,scan):
    if a > scan.shape[1]:
        a = int(scan.shape[1])
    return int(np.around(a))
def right_bound_z(a,scan):
    if a > scan.shape[2]:
        a = int(scan.shape[2])
    return int(np.around(a))

def mesh_volume(faces,verts):
    volume = 0
    for face in faces:
        volume += signed_vol(verts[face[0]],verts[face[1]],verts[face[2]])
    return abs(volume)

def signed_vol(p1,p2,p3):
    v321 = p3[0]*p2[1]*p1[2]
    v231 = p2[0]*p3[1]*p1[2]
    v312 = p3[0]*p1[1]*p2[2]
    v132 = p1[0]*p3[1]*p2[2]
    v213 = p2[0]*p1[1]*p3[2]
    v123 = p1[0]*p2[1]*p3[2]
    return (1.0/6.0)*(-v321 + v231 + v312 - v132 - v213 + v123)

loc = find_objects(labeled_array)
print (loc)
boxsize = 1 # Width of Additional Layers of Voxels around bounding box used for lacunar morphology calculation
scan = seg_vol # It will be used in the future just to take the reference image dimension

for x in range(3): #num_features_new is equal to the number of found lacunae

    xmin_temp = loc[x][0].start
    xmax_temp = loc[x][0].stop
    ymin_temp = loc[x][1].start
    ymax_temp = loc[x][1].stop
    zmin_temp = loc[x][2].start
    zmax_temp = loc[x][2].stop
    print (xmin_temp, xmax_temp, ymin_temp, ymax_temp, zmin_temp, zmax_temp )

    xmin = left_bound(xmin_temp-boxsize)
    xmax = right_bound_x(xmax_temp+boxsize,scan)
    ymin = left_bound(ymin_temp-boxsize)
    ymax = right_bound_y(ymax_temp+boxsize,scan)
    zmin = left_bound(zmin_temp-boxsize)
    zmax = right_bound_z(zmax_temp+boxsize,scan)

    vol = np.copy(labeled_array[xmin:xmax,ymin:ymax,zmin:zmax])
    
    vol[vol != (x+1)] = 0 # To make sure we only have one lacuna inside the box


    # Find surfaces in 3d volumetric data
    vol_verts, vol_faces, vol_normals, vol_values = measure.marching_cubes(vol, method='lewiner')
    # Compute surface area, given vertices and triangular faces
    est_surface = skimage.measure.mesh_surface_area(vol_verts, vol_faces)
    SurfaceArea.append(est_surface)

    # Compute volume
    est_volume = mesh_volume(vol_faces,vol_verts)
    Volume.append(est_volume)

    regionprops = skimage.measure.regionprops(vol)

    for region in regionprops:
        Area_temp = region.area 
        Area.append(Area_temp)
        Centroid_temp = np.array(region.centroid)
        distance_matrix_x = np.array(abs(vol_verts[:,0]- Centroid_temp[0]))
        distance_matrix_y = np.array(abs(vol_verts[:,1]- Centroid_temp[1]))
        distance_matrix_z = np.array(abs(vol_verts[:,2]- Centroid_temp[2]))
        len_temp = (((np.amax(distance_matrix_x) ** 3) + (np.amax(distance_matrix_y) ** 3) + (np.amax(distance_matrix_z) ** 3)) ** (1/3))
        Len.append(len_temp)
        Centroid_temp[0] = Centroid_temp[0] + xmin
        Centroid_temp[1] = Centroid_temp[1] + ymin
        Centroid_temp[2] = Centroid_temp[2] + zmin
        Centroid.append(Centroid_temp)

    # LACUNAE STRETCH PARAMETER
    ## Principle Component Analysis
    vol[vol == x+1] = 1
    vol_pca_test_evals, vol_pca_test_evecs = compute_initial_guess.get_principal_axes_grayscale(vol,0.5)
    vector_Lc_Or1_x = np.append(vector_Lc_Or1_x,vol_pca_test_evecs[0,2])
    vector_Lc_Or1_y = np.append(vector_Lc_Or1_y,vol_pca_test_evecs[1,2])
    vector_Lc_Or1_z = np.append(vector_Lc_Or1_z,vol_pca_test_evecs[2,2])
    ## Find lacunae stretch parameters starting from eigenvaluew just found. Eigenvalues are sorted from the biggest to the lowest
    res_St = (((vol_pca_test_evals[0])**(1/2))-((vol_pca_test_evals[2])**(1/2)))/((vol_pca_test_evals[0])**(1/2))
    Lc_St = np.append(Lc_St,res_St) 

    # LACUNAE OBLATENESS PARAMETER
    res_Ob = 2*((((vol_pca_test_evals[1])**(1/2))-((vol_pca_test_evals[2])**(1/2)))/(((vol_pca_test_evals[0])**(1/2))-((vol_pca_test_evals[2])**(1/2))))-1
    Lc_Ob = np.append(Lc_Ob,res_Ob) 
    
    # LACUNAE ORIENTATION
    MainAxisVector = np.append(MainAxisVector, np.reshape(vol_pca_test_evecs[:,0], (1,3)), axis = 0)
    #LcLe=np.append(LcLe,np.reshape(vol_pca_test_evals/(2*(5**(1/2))),(1,3)), axis = 0)  #find lacuna length and I append he new value in the array LcLe
    ## Lacuna orientation 1
    Lc_Or_res_1_x= vol_pca_test_evecs[0,0]
    Lc_Or_res_1_y= vol_pca_test_evecs[1,0]
    Lc_Or_res_1_z= vol_pca_test_evecs[2,0]
    Lc_Or_1_x = np.append(Lc_Or_1_x,Lc_Or_res_1_x) 
    Lc_Or_1_y = np.append(Lc_Or_1_y,Lc_Or_res_1_y) 
    Lc_Or_1_z = np.append(Lc_Or_1_z,Lc_Or_res_1_z) 
    ## Lacuna orientation 2
    Lc_Or_res_2_x= vol_pca_test_evecs[0,1]
    Lc_Or_res_2_y= vol_pca_test_evecs[1,1]
    Lc_Or_res_2_z= vol_pca_test_evecs[2,1]
    Lc_Or_2_x = np.append(Lc_Or_2_x,Lc_Or_res_2_x) 
    Lc_Or_2_y = np.append(Lc_Or_2_y,Lc_Or_res_2_y) 
    Lc_Or_2_z = np.append(Lc_Or_2_z,Lc_Or_res_2_z) 
    ## Max distance
    distance_max_temp = vol_pca_test_evals[0]
    distance_min_temp = vol_pca_test_evals[2]
    Distance_Max.append(distance_max_temp/2)
    Distance_Min.append(distance_min_temp)

# From regionprops and PCA to calculate further properties:
Distance_Max = np.array(Distance_Max)
Distance_Min = np.array(Distance_Min)

# Convert lacuna volume from voxels to cubic micrometers
lac_vol_micron = np.array(Area)
lac_Vol_cubic_um = lac_vol_micron * ((voxelsize*1000) ** 3)

# SURFACE TO VOLUME RATIO (from mesh)
SV_ratio = []
for a in range(len(Volume)):
    SV_ratio.append(SurfaceArea[a]/Volume[a])

# SURFACE AREA in micrometers
SA_um = []
for s in SurfaceArea:
    SA_um.append(s * ((voxelsize.magnitude*1000) ** 2))

# SURFACE TO VOLUME RATIO calculated from um values
LcSAV_ratio = []
for a in range(len(Volume)):
    LcSAV_ratio.append(SA_um[a]/lac_Vol_cubic_um.magnitude[a])

# Minor/Major axis ratio
MM_Ratio = Distance_Min/Distance_Max

# LACUNAR LENGTH in um (assuming symetric shape)
Len = np.array(Len)
Length_um = Len * (voxelsize.magnitude*1000) * 2

# Convert Max and Min to um
Distance_Max = Distance_Max * (voxelsize.magnitude*1000) 
Distance_Min = Distance_Min * (voxelsize.magnitude*1000) 

# Lacunar ID
lac_id = []
for a in range(len(Volume)):
    lac_id.append(a+1)

# Total Lacunar Density [lac per mm^3]
total_vol_lac = np.sum(lac_vol_micron)
total_vol_bone = scan.shape[0]*scan.shape[1]*scan.shape[2] * ((voxelsize*1000) ** 3)
lac_density = np.amax(lac_id)/total_vol_bone

# Volume of Lacunar Density [volume fraction of lacunae in bone]
lac_density_vol = total_vol_lac/total_vol_bone

# Correction since Centroid is always off by -1 in each direction
Centroid = np.array(Centroid)
Centroid += 1








