import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=0):

	# apply filter kernel
	kernel = createKernel(kernelSize, sigma, theta)
	 #np.array(labels))
	imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
	(_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	imgThres = 255 - imgThres

	# find connected components. OpenCV: return type differs between OpenCV2 and 3
	if cv2.__version__.startswith('3.'):
		(_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	else:
		(components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# append components to result
	res = []
	for c in components:
		# skip small word candidates
		if cv2.contourArea(c) < minArea:
			continue
		# append bounding box and image of word to result list
		currBox = cv2.boundingRect(c) # returns (x, y, w, h)
		(x, y, w, h) = currBox
		currImg = img[y:y+h, x:x+w]
		res.append((currBox, currImg))

	# return list of words, sorted by x-coordinate
	return sorted(res, key=lambda entry:entry[0][0])


def prepareImg(img, height):
	"""convert given image to grayscale image (if needed) and resize to desired height"""
	assert img.ndim in (2, 3)
	if img.ndim == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	h = img.shape[0]
	factor = height / h
	return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def createKernel(kernelSize, sigma, theta):
	"""create anisotropic filter kernel according to given parameters"""
	assert kernelSize % 2 # must be odd size
	halfSize = kernelSize // 2
	
	kernel = np.zeros([kernelSize, kernelSize])
	sigmaX = sigma
	sigmaY = sigma * theta
	
	for i in range(kernelSize):
		for j in range(kernelSize):
			x = i - halfSize
			y = j - halfSize
			
			expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
			xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
			yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
			
			kernel[i, j] = (xTerm + yTerm) * expTerm

	kernel = kernel / np.sum(kernel)
	return kernel

#wordSegmentation('C:\\Users\\PIN\\Desktop\\Handwriting-OCR-master\\Resource\\test_img\\9.png', kernelSize=25, sigma=11, theta=7, minArea=0)
test_img = '/home/lampe/Bureau/Handwriting-OCR/Resource/test_img/2.png'
img = prepareImg(cv2.imread(test_img), 64)
img2 = img.copy()
res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
if not os.path.exists('tmp'):
	os.mkdir('tmp')

for (j, w) in enumerate(res):
	(wordBox, wordImg) = w
	(x, y, w, h) = wordBox
	cv2.imwrite('tmp/%d.png'%j, wordImg)
	cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),1) # draw bounding box in summary imag
cv2.imwrite('Resource/summary.png', img2)
#plt.imshow(img2)
imgFiles = os.listdir('tmp')
imgFiles = sorted(imgFiles)
#plt.show()
