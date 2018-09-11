from mnist import MNIST
from numpy import *
from pprint import pprint
from collections import defaultdict
import random  
from math import sqrt
import matplotlib.pyplot as plt
import operator


def pointAvg(points):
	dimensions = len(points[0])

	newCenters = []

	for dimension in xrange(dimensions):
		dimSum = 0
		for p in points:
			dimSum += p[dimension]

		newCenters.append(dimSum / float(len(points)))

	return newCenters


def generateK(mat_data, K): 
#generate the initial K point from the current images (points). Random choose a image as center.
	centers = []
	rand_index = []

	while len(centers) < K:
	 	rand = random.randint(0,len(mat_data))

	 	if rand not in rand_index:
	 		rand_index.append(rand)
	 		# centers.append(mat_data[rand].tolist())
	 		centers.append(mat_data[rand])
	return centers

def distance(a, b):
	dimensions = len(a)
	total = 0

	for d in xrange(dimensions):
		diff_sq = (a[d] - b[d]) ** 2
		total +=  diff_sq
	return sqrt(total)

def assignPoints(mat_data, centers): #assign each image to the nearest point.
	assignments = []
	for image in mat_data:
		shortest = ()
		shortest_index = 0;
		for c in xrange(len(centers)):
		 	val = distance(image, centers[c])
		 	if val < shortest:
		 		shortest = val
		 		shortest_index = c
 		assignments.append(shortest_index)
	return assignments

def updateCenter(mat_data, assignments):
	new_mean = defaultdict(list)
	centers = []

	for assignment, point in zip(assignments, mat_data):
		new_mean[assignment].append(point)

	for points in new_mean.itervalues():
		centers.append(pointAvg(points))

	return centers

def Kmeans(mat_data, K):
	k_point = generateK(mat_data, K)
	assignments = assignPoints(mat_data, k_point)

	old_assignments = None
	i = 0 
	new_centers = None
	while assignments != old_assignments:
		i = i + 1
		new_centers = updateCenter(mat_data,assignments)

		old_assignments = assignments
		assignments = assignPoints(mat_data, new_centers)

	return new_centers
		

def getDataSet(data_amount):
	# For global PCA
	mndata = MNIST('sample') #sample is the directory of the MNIST
	images, labels = mndata.load_training() #image is 28*28 image, labels is the number which the image presents
	mat_training = zeros((data_amount,28*28)) #initialize the matrix size
	lbl_training = [] 

	for i in range(data_amount) : #get the data_amount of the data
		mat_training[i] = images[i]
		lbl_training.append(labels[i])

	return mat_training, lbl_training

def drawXY(img, x, y, i):
	plt.subplot(x,y,i)
	plt.imshow(array(img).reshape(28,28), cmap='gray')
	plt.axis('off')
	
knum = 20
mat_training, lbl_training = getDataSet(5000)
kmeanMat = Kmeans(mat_training, knum)

for i in range(knum):
	drawXY(kmeanMat[i], 4, 5, i+1)

plt.show()

