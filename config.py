import numpy as np

import constants


targets = {}

center = constants.GRID_SIZE//2

# Initial grid
init = np.zeros((constants.GRID_SIZE, constants.GRID_SIZE))
init[center,center]=1

# Rectangle
rect = np.zeros((constants.GRID_SIZE, constants.GRID_SIZE))
width = constants.GRID_SIZE//3 * 2
height = constants.GRID_SIZE//3
rect[center-height//2:center+height//2,center-width//2:center+width//2+1] = 1
targets['rectangle'] = rect

# Square
square = np.zeros((constants.GRID_SIZE, constants.GRID_SIZE))
length = constants.GRID_SIZE//2
square[center-length//2:center+length//2+1, center-length//2:center+length//2+1]=1
targets['square']=square

# Circle
circle = np.zeros((constants.GRID_SIZE, constants.GRID_SIZE))
radius = constants.GRID_SIZE//4
r2 = np.arange(-center, center + 1) ** 2 if constants.GRID_SIZE % 2==1 else np.arange(-center, center) ** 2
dist2 = r2[:, None] + r2
circle[dist2 < radius**2] = 1
targets['circle'] = circle

# Xenobot
xenobot = np.zeros((25, 25))
xenobot[9:16, 6:19] = 1
xenobot[15:21, 6:10] = 1
xenobot[15:21, 15:19] = 1
xenobot[9,6:8]=0
xenobot[9, 17:19]=0
xenobot[10, 6:19:12] = 0
xenobot[20,6:8]=0
xenobot[20, 17:19] = 0
xenobot[19, 6:19:12] = 0
xenobot[18, 6:19:12] = 0
xenobot[20, 9:16:6] = 0
xenobot[17, 10:15:4]=1
xenobot[16, 10:12] = 1
xenobot[16, 13:15] = 1
targets['xenobot'] = xenobot

# Biped 
biped = np.zeros((constants.GRID_SIZE,constants.GRID_SIZE))
length = constants.GRID_SIZE//2
biped[center-length//2:center+length//2+1, center-length//2:center+length//2+1]=1
biped[center:center+length//2+1, center-length//4:center+length//4+1] = 0
targets['biped']=biped

# Triangle
triangle = np.zeros((constants.GRID_SIZE, constants.GRID_SIZE))

i=0
# if x is between center-length//2:center+length//2 (for)
# y goes from center-i:center+i+1
for x in range(center-length//2,center+length//2):
    triangle[x,center-i:center+i+1]=1
    i+=1
targets['triangle']=triangle
