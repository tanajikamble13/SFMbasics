#loading needed libraries 
import utils as ut 
import SfM as sfmnp
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

import cv2 
import numpy as np 
import math
lc=[]
rc=[]


f_inv = 298.257224
#f = 1.0 / f_inv
f = 0.0 # here we considered ideal earth shape
e2 = 1 - (1 - f) * (1 - f)
altitude=4.281411
prince_gps=[41.333317727121900,-74.190944277933500,41.333354457942900,-74.191021643789500]

def get_cartesian(latitude=None,longitude=None,altitude=0.0):
    R = 6378137 #radius in meter
    #R = 6371000
    cosLat = math.cos(latitude * math.pi / 180)
    sinLat = math.sin(latitude * math.pi / 180)

    cosLong = math.cos(longitude * math.pi / 180)
    sinLong = math.sin(longitude * math.pi / 180)

    c = 1 / math.sqrt(cosLat * cosLat + (1 - f) * (1 - f) * sinLat * sinLat)
    s = (1 - f) * (1 - f) * c

    x = (R*c + altitude) * cosLat * cosLong
    y = (R*c + altitude) * cosLat * sinLong
    z = (R*s + altitude) * sinLat
    return [x,y,z]

def get_latlon(X=None,Y=None,Z=None):
    R = 6378137
    #R = 6371000
    lat = math.degrees(math.asin(Z/R))
    lon = math.degrees(math.atan2(Y,X))
    return[lat,lon]

roll = 0.0271091714
pitch =0.0188595763
yaw = 3.137739087
YPR=[roll,pitch,yaw]

def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    RotationMat = np.dot(R_z, np.dot( R_y, R_x ))
 
    return RotationMat

R=eulerAnglesToRotationMatrix(YPR)
print ("Rotation matrix \n", R)




#Reading two images for reference
img1 = cv2.imread('testL.jpg')
img2 = cv2.imread('testR.jpg')

#Converting from BGR to RGB format
img1 = img1[:,:,::-1]
img2 = img2[:,:,::-1]

#NOTE: you can adjust appropriate figure size according to the size of your screen
fig,ax=plt.subplots(ncols=2,figsize=(9,4)) 
ax[0].imshow(img1)
ax[1].imshow(img2)

#NOTE: you can adjust appropriate figure size according to the size of your screen
fig,ax=plt.subplots(ncols=2,figsize=(9,4)) 
ax[0].imshow(img1)
ax[1].imshow(img2)
plt.show()

#Getting SIFT/SURF features for image matching (this might take a while)
kp1,desc1,kp2,desc2,matches=ut.GetImageMatches(img1,img2)

#Aligning two keypoint vectors
img1pts,img2pts,img1idx,img2idx=ut.GetAlignedMatches(kp1,desc1,kp2,desc2,matches)

img1pts_, img2pts_ = img1pts[:8], img2pts[:8]
print("image 1 points",img1pts[:8].shape)
'''
#Fundamental Matrix 8 point algorithm
Fgt, mask = cv2.findFundamentalMat(img1pts_,img2pts_,method=cv2.FM_8POINT)
#fundamental smnf Method
F = sfmnp.EstimateFundamentalMatrix(img1pts_,img2pts_)
#smnf normalized 8 point algo
F_normalized = sfmnp.EstimateFundamentalMatrixNormalized(img1pts_,img2pts_)
'''
# Fundamental matrix with RANSAC based method

#Fgt, maskgt = cv2.findFundamentalMat(img1pts,img2pts,method=cv2.FM_RANSAC,)
#maskgt = maskgt.astype(bool).flatten()

#F, mask = sfmnp.EstimateFundamentalMatrixRANSAC(img1pts,img2pts,.1,iters=20000)

F, mask = cv2.findFundamentalMat(img1pts,img2pts,method=cv2.FM_RANSAC,)
mask = mask.astype(bool).flatten()

#Epipolar Lines Computation
lines2=sfmnp.ComputeEpiline(img1pts[mask],1,F)
lines1=sfmnp.ComputeEpiline(img2pts[mask],2,F)

# Visualization Epipolar Lines
tup = ut.drawlines(img2,img1,lines2,img2pts[mask],img1pts[mask],drawOnly=10,
                   linesize=10,circlesize=30)
epilines2 = np.concatenate(tup[::-1],axis=1) #reversing the order of left and right images
plt.figure(figsize=(9,4))
plt.imshow(epilines2)
tup = ut.drawlines(img1,img2,lines1,img1pts[mask],img2pts[mask],drawOnly=10,
                   linesize=10,circlesize=30)
epilines1 = np.concatenate(tup,axis=1) 
plt.figure(figsize=(9,4))
plt.imshow(epilines1)
plt.show()

fx=1761.320022138
prince_pt=[969.85441709, 565.51099872]
K = np.array([[fx,   0.00000000, prince_pt[0]], 
                 [  0.00000000, fx, prince_pt[1]],
                 [  0.00000000,   0.00000000,   1.00000000]])
E = K.T.dot(F.dot(K))

R1,R2,t = sfmnp.ExtractCameraPoses(E)
t = t[:,np.newaxis]

# Essential Matrix calculation
E = K.T.dot(F.dot(K))
R1,R2,t = sfmnp.ExtractCameraPoses(E)
t = t[:,np.newaxis]

#Triangulation: DLT Method
def GetTriangulatedPts(img1pts,img2pts,K,R,t,triangulateFunc): 
    img1ptsHom = cv2.convertPointsToHomogeneous(img1pts)[:,0,:]
    img2ptsHom = cv2.convertPointsToHomogeneous(img2pts)[:,0,:]

    img1ptsNorm = (np.linalg.inv(K).dot(img1ptsHom.T)).T
    img2ptsNorm = (np.linalg.inv(K).dot(img2ptsHom.T)).T

    img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:,0,:]
    img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:,0,:]
    
    pts4d = triangulateFunc(np.eye(3,4),np.hstack((R,t)),img1ptsNorm.T,img2ptsNorm.T)
    pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:,0,:]
    
    return pts3d

pts3dgt = GetTriangulatedPts(img1pts[mask],img2pts[mask],K,R2,t,cv2.triangulatePoints)
pts3d = GetTriangulatedPts(img1pts[mask],img2pts[mask],K,R2,t,sfmnp.Triangulate)
print("Points in 3D",pts3d[:5])



def click_Left(eventL, x, y, flags, param):
# grab references to the global variables
    global refPtL, cropping
    #refPt=[[0,0]]
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    
    if eventL == cv2.EVENT_LBUTTONDOWN:
        refPtL = [[x, y]]
        cropping = True
        cv2.circle(imageL,(refPtL[0][0], refPtL[0][1]), 5, (0,0,255), -1)
        cv2.imshow("Left image", imageL)
        #print("Event locked")
 
    # check to see if the left mouse button was released
    elif eventL == cv2.EVENT_LBUTTONUP:
        lc.append([x, y])
        print("lc in loop",lc)
    # record the ending (x, y) coordinates and indicate that
    # the cropping operation is finished
        #refPt.append((x, y))
    return 0


def click_Right(eventR, a, b, flags, param):
# grab references to the global variables
    global ix,iy
    #refPt=[[0,0]]
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    
    if eventR == cv2.EVENT_LBUTTONDOWN:
        refPtR = [[a, b]]
        cv2.circle(imageR,(refPtR[0][0], refPtR[0][1]), 5, (255,0,0), -1)
        cv2.imshow("Right image", imageR)
    # check to see if the left mouse button was released
    elif eventR == cv2.EVENT_LBUTTONUP:
        rc.append([a, b])
        print("rc in loop",rc)  
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        #refPt.append((x, y))
    return 0

# load the image, clone it, and setup the mouse callback function
imageL = cv2.imread("testL.jpg")
'''
#imageL = cv2.undistortPoints(imageL, camera_matrix, dist_coeffs)
h,  w = image1.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs,(w,h),1,(w,h)) #alpha=1 means black holes
iml = cv2.undistort(image1,camera_matrix, dist_coeffs, None, newcameramtx)
# crop the image
x,y,w,h = roi
imageL = iml[y:y+h, x:x+w]
'''
#print("Image Shape ",imageL.shape) 
cv2.namedWindow("Left image",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Left image", click_Left)

imageR = cv2.imread("testR.jpg")
#imr = cv2.undistort(image2,camera_matrix, dist_coeffs, None, newcameramtx)
#imageR = imr[y:y+h, x:x+w]
#print("Image Shape ",imageL.shape) 
cv2.namedWindow("Right image",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Right image", click_Right)

#print("refPtL",click_Left)
while True:
    # display the image and wait for a keypress
    cv2.imshow("Left image", imageL)
    cv2.imshow("Right image", imageR)
    key = cv2.waitKey(1) & 0xFF
 
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()
 
    # if the 'd' key is pressed, break from the loop
    elif key == ord("d"):
        
        break
cv2.destroyAllWindows()

pts3dgt2 = GetTriangulatedPts(np.array(lc).reshape(-1,2),np.array(rc).reshape(-1,2),K,R2,t,cv2.triangulatePoints)
pts3d2 = GetTriangulatedPts(np.array(lc).reshape(-1,2),np.array(rc).reshape(-1,2),K,R2,t,sfmnp.Triangulate)
print("Points in 3D",pts3d2.shape)

#leftcamCart=get_cartesian(prince_gps[0],prince_gps[1])
#rightcamCart=get_cartesian(prince_gps[2],prince_gps[3])
#rightcamCartMod=[0,0,0]
#rightcamCartMod[0]=rightcamCart[0]-leftcamCart[0]
#rightcamCartMod[1]=rightcamCart[1]-leftcamCart[1]
#rightcamCartMod[2]=rightcamCart[2]-leftcamCart[2]


k=1.5708
stereo_Y =  R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(k), -math.sin(k) ],
                    [0,         math.sin(k), math.cos(k)  ]
                    ])
PixArray=np.array([[pts3d2[0][0]],[pts3d2[0][1]],[pts3d2[0][2]]])
rotatedCord=np.dot(np.linalg.inv(stereo_Y),PixArray)
pix3DFinal=np.dot(np.linalg.inv(R),PixArray)


cart=get_cartesian(prince_gps[0],prince_gps[1])
X=cart[0]-pix3DFinal[0][0]
Y=cart[1]-pix3DFinal[2][0]
Z=cart[2]-pix3DFinal[1][0]
print("\n Vehicle GPS : ",prince_gps[0],prince_gps[1])
print("Pixel (%d,%d) "% (lc[0][0],lc[0][1]),  " Lat-long YPR", get_latlon(X,Y,Z))
#print("Pixel (%d,%d) "% (lc[0][0],lc[0][1]),  " Lat-long Standar ", get_latlon(X2,Y2,Z2))




