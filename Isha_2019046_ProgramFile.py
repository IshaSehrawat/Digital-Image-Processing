import numpy
import cv2
from math import *

# # Question 3 Solution -----------------------------------------------------------------------------
print('Solution to 3 goes here')
input=cv2.imread('./Isha_2019046_Input3.bmp',0)
# x and y store the dimensions of input image
x,y = input.shape
# f is the interpolation factor
f = 0.5
print('Interpolation Factor :',f)
# f = ceil(f)
# to handle case where f is decimal
# p and q store the dimensions of output image
p,q = ceil(x*f),ceil(y*f)
# output matrix to store the image pixel values default -1
output=numpy.ones((p,q))*-1

# filling all the indices from the input
# which dont need to be computed using bilinear
for i in range(x):
    for j in range(y):
        output[ceil(i*f)][ceil(j*f)] = input[i][j]

# Filling all but leaving last row and column
for i in range(p-1):
    for j in range(q-1):
        if(output[i][j] == -1):
            # m and n store coordinate mapping to i,j in input
            m,n = i/f,j/f

            # Logic to find out 4 nearest neighbors
            if(m==0):
                x1,x2 = 0,1
            elif(m == ceil(m)):
                x1,x2 = m-1,m
            else:
                x1=floor(m)
                x2=ceil(m)

            if(n==0):
                y1,y2 = 0,1
            elif(n == ceil(n)):
                y1,y2 = n-1,n
            else:
                y1=floor(n)
                y2=ceil(n)

            # V=XA
            # X is the matrix of the four points
            # V has the pixel value in the input at those 4 points
            x1,x2,y1,y2=int(x1),int(x2),int(y1),int(y2)
            # Clamping values to take care of singularity of matrix
            x1 = min(x1,x-1)
            x2 = min(x2,x-1)
            y1 = min(y1,y-1)
            y2 = min(y2,y-1)
            if(x1 == x2):
                x2 = x1-1
            if(y1 == y2):
                y2 = y1-1
            X = [[x1,y1,x1*y1,1],[x2,y2,x2*y2,1],[x1,y2,x1*y2,1],[x2,y1,x2*y1,1]]
            V = [[input[x1][y1]],[input[x2][y2]],[input[x1][y2]],[input[x2][y1]]]
            # A=inv(X).V
            A = numpy.dot(numpy.linalg.inv(X),V)
            # Find the pixel value at i,j using m,n
            bruh = int(numpy.dot(numpy.array([m,n,m*n,1]),A))
            output[i][j] = round(bruh)
            # clamp at 255
            if(output[i][j] > 255):
                output[i][j] = 255

# Type Conversion
output = output.astype(numpy.uint8)
# Copying values for last column and row from previous ones
for i in range(p-1):
    for j in range(q-1,len(output[0])):
        output[i][j] = output[i][j-1]

for j in range(len(output[0])):
    for i in range(q-1,len(output)):
        output[i][j] = output[i-1][j]

# Filling for last cell
output[p-1][q-1] = output[p-2][q-1]
cv2.imshow("Input",input)
cv2.imshow("Output",output)
cv2.waitKey(0)
cv2.destroyAllWindows()
print()

# Question 4 Solution -----------------------------------------------------------------------------
print('Solution to 4 goes here')

Input=cv2.imread("./Isha_2019046_Input4.jpg",0)
# x and y store the dimensions of Input image
x,y = Input.shape
# InputCanvas to show the input image with respect to the new origin
InputCanvas = numpy.ones((1200,1265),numpy.uint8)*0
Output = numpy.ones((1200,1200),numpy.uint8)*-1

# origin at (100,100) = oo
oo = 150
print('Origin :',oo,',',oo)
# Fill Input image onto campus origin at oo,oo
for i in range(x):
    for j in range(y):
        InputCanvas[oo+i][oo+j] = Input[i][j]

# T1 is tranformation matrix for rotation by 45 deg
# T2 is tranformation matrix for scaling by 2
# T3 is tranformation matrix for translation by 30 pixel on x and y
T1 = [[0.70710678,-0.70710678,0],[0.70710678,0.70710678,0],[0,0,1]]
T2 = [[2,0,0],[0,2,0],[0,0,2]]
T3 = [[1,0,0],[0,1,0],[30,30,1]]
# Tx = T1.T2.T3
Tx = numpy.dot(numpy.dot(T1,T2),T3)
print('Transformation Matrix :')
print(Tx)
# T4 is to translate at the origin at oo,oo
T4 = [[1,0,0],[0,1,0],[oo,oo,1]]
# T is the final transformation matrix
T = numpy.dot(Tx,T4)

# For optimazation calculate max coordinate of the output image
xm,ym = 0,0
for i in range(x):
    for j in range(y):
        O = numpy.dot(numpy.array([i,j,1]),T)
        xm = max(xm,O[0])
        ym = max(ym,O[1])
xm,ym = int(xm),int(ym)
# print(xm,ym)
for i in range(xm+1):
    for j in range(ym+1):
        if(Output[i][j] == -1):
            # m and n store coordinate mapping to i,j in Input
            O = numpy.dot(numpy.array([i,j,1]),numpy.linalg.inv(T))
            m,n = O[0],O[1]
            # If the mapping is not required continue
            if(m < 0 or ceil(m) > x):
                continue
            if(n < 0 or ceil(n) > y):
                continue

            if(m == 0):
                x1,x2 = 0,1
            elif(m == ceil(m)):
                x1,x2 = m-1,m
            else:
                x1 = floor(m)
                x2 = ceil(m)

            if(n == 0):
                y1,y2 = 0,1
            elif(n == ceil(n)):
                y1,y2 = n-1,n
            else:
                y1 = floor(n)
                y2 = ceil(n)

            # V=XA
            # X is the matrix of the four points
            # V has the pixel value in the Input at those 4 points
            x1,x2,y1,y2=int(x1),int(x2),int(y1),int(y2)
            x1 = min(x1,x-1)
            x2 = min(x2,x-1)
            y1 = min(y1,y-1)
            y2 = min(y2,y-1)
            if(x1 == x2):
                x1 = x2-1
            if(y1 == y2):
                y1 = y2-1

            X = [[x1,y1,x1*y1,1],[x2,y2,x2*y2,1],[x1,y2,x1*y2,1],[x2,y1,x2*y1,1]]
            V = [[Input[x1][y1]],[Input[x2][y2]],[Input[x1][y2]],[Input[x2][y1]]]
            # A=inv(X).V
            A = numpy.dot(numpy.linalg.inv(X),V)

            bruh = int(numpy.dot(numpy.array([m,n,m*n,1]),A))
            Output[i][j] = round(bruh)
            if(Output[i][j] > 255):
                Output[i][j] = 255

# Paint the rest of canvas black
for i in range(len(Output)):
    for j in range(len(Output[0])):
        if(Output[i][j] == -1):
            Output[i][j] = 0

InputCanvas = InputCanvas.astype(numpy.uint8)
Output = Output.astype(numpy.uint8)

# Marking the origin at oo,oo as white for reference
InputCanvas[oo][oo] = 255
Output[oo][oo] = 255
cv2.imshow("Input",InputCanvas)
cv2.imshow("Output",Output)
# cv2.imwrite(r"C:\Users\Asus\Desktop\Python Files\Unregistered.png",Output)
cv2.waitKey(0)
cv2.destroyAllWindows()
print()

Question 5 Solution -----------------------------------------------------------------------------
print('Solution to 5 goes here')

ref_img = cv2.imread('./Isha_2019046_Input4.jpg', 0)
unreg_img = cv2.imread('./Isha_2019046_UnReg5.png', 0)
cv2.imshow('Reference Image', ref_img)
cv2.imshow('Unregistered Image',unreg_img)

# X contains the corresponding points from unregistered image
# V contains the corresponding points from reference image
X = [[266,152,1],[271,146,1],[240,141,1]]
V = [[40,20,1],[44,20,1],[34,8,1]]
T = numpy.dot(numpy.linalg.inv(V),X)
Output = numpy.ones((2845,2178),numpy.uint8)*-1

print('Transformation Matrix used was:')
print(T)
x,y = unreg_img.shape
# To optimize for only the maximum coordinate of output image
xm,ym = 0,0
for i in range(x):
    for j in range(y):
        O = numpy.dot(numpy.array([i,j,1]),T)
        xm = max(xm,O[0])
        ym = max(ym,O[1])
xm,ym = int(xm),int(ym)
for i in range(xm+1):
    for j in range(ym+1):
        if(Output[i][j] == -1):
            # m and n store coordinate mapping to i,j in Input
            O = numpy.dot(numpy.array([i,j,1]),T)
            m,n = O[0],O[1]
            if(m < 0 or ceil(m) > x):
                continue
            if(n < 0 or ceil(n) > y):
                continue

			# Nearest neighbors logic
            if(m == 0):
                x1,x2 = 0,1
            elif(m == ceil(m)):
                x1,x2 = m-1,m
            else:
                x1 = floor(m)
                x2 = ceil(m)

            if(n == 0):
                y1,y2 = 0,1
            elif(n == ceil(n)):
                y1,y2 = n-1,n
            else:
                y1 = floor(n)
                y2 = ceil(n)

            # V=XA
            # X is the matrix of the four points
            # V has the pixel value in the Input at those 4 points
            x1,x2,y1,y2=int(x1),int(x2),int(y1),int(y2)
            x1 = min(x1,x-1)
            x2 = min(x2,x-1)
            y1 = min(y1,y-1)
            y2 = min(y2,y-1)
            if(x1 == x2):
                x1 = x2-1
            if(y1 == y2):
                y1 = y2-1

            X = [[x1,y1,x1*y1,1],[x2,y2,x2*y2,1],[x1,y2,x1*y2,1],[x2,y1,x2*y1,1]]
            V = [[unreg_img[x1][y1]],[unreg_img[x2][y2]],[unreg_img[x1][y2]],[unreg_img[x2][y1]]]
            # A=inv(X).V
            A = numpy.dot(numpy.linalg.inv(X),V)

            bruh = int(numpy.dot(numpy.array([m,n,m*n,1]),A))
            Output[i][j] = round(bruh)
            if(Output[i][j] > 255):
                Output[i][j] = 255


# Using Forward Mapping
# X = VT
# for i in range(len(unreg_img)):
# 	for j in range(len(unreg_img[0])):
# 		O = numpy.dot(numpy.array([i,j,1]),numpy.linalg.inv(T))
# 		m,n = O[0],O[1]
# 		m,n = int(m),int(n)
# 		# print(m,n,i,j)
# 		if(m<0 or n<0 or m>=650 or n>=1200):
# 			continue
# 		Output[m][n] = unreg_img[i][j]

for i in range(len(Output)):
	for j in range(len(Output[0])):
		if(Output[i][j] == -1):
			Output[i][j] = 0

Output = Output.astype(numpy.uint8)
cv2.imshow("Registered Image",Output)
cv2.waitKey(0)
cv2.destroyAllWindows()
