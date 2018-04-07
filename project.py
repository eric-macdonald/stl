import doctest
from euclid import *
import stl
from stl import mesh
import math
import cv2
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import sys

ztrans = float(sys.argv[1])
anglex = sys.argv[2]
angley = sys.argv[3]
anglez = sys.argv[4]
filename = sys.argv[5]

print "ztranslation " + str(ztrans)
print "angleX " + str(anglex)
print "angleY " + str(angley)
print "angleZ " + str(anglez)
print "filename " + str(filename)


height = 800
width = 800 
print "width and height " + str(height) + " " + str(width)
fov   = float(math.pi / 4)
print "fov " + str(fov)
print "fov " + str(fov)
aspect = float(width)/float(height)
adderx = float(width)/float(2)
addery = float(height)/float(2)
znear = float(1)
zfar = float(100) 



def find_mins_maxs(obj):
    minx = maxx = miny = maxy = minz = maxz = None
    for p in obj.points:
    # p contains (x, y, z)
        if minx is None:
            minx = p[stl.Dimension.X]
            maxx = p[stl.Dimension.X]
            miny = p[stl.Dimension.Y]
            maxy = p[stl.Dimension.Y]
            minz = p[stl.Dimension.Z]
            maxz = p[stl.Dimension.Z]
        else:
            maxx = max(p[stl.Dimension.X], maxx)
            minx = min(p[stl.Dimension.X], minx)
            maxy = max(p[stl.Dimension.Y], maxy)
            miny = min(p[stl.Dimension.Y], miny)
            maxz = max(p[stl.Dimension.Z], maxz)
            minz = min(p[stl.Dimension.Z], minz)
    return minx, maxx, miny, maxy, minz, maxz

convertx = float(float(anglex)*(math.pi/180))
converty = float(float(angley)*(math.pi/180))
convertz = float(float(anglez)*(math.pi/180))

print filename
readmesh = mesh.Mesh.from_file(filename)
readmesh.rotate([1,0,0], convertx)
readmesh.rotate([0,1,0], converty)
readmesh.rotate([0,0,1], convertz)
readmesh.z += -1*ztrans 
#print "max " + str(readmesh.max_)
#print "min " + str(readmesh.min_)
minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(readmesh)
#print "min X " + str(minx)
#print "max X" + str(maxx)
#print "min Y " + str(miny)
#print "max Y" + str(maxy)
#print "min Z " + str(minz)
#print "max Z" + str(maxz)

project = Matrix4.new_perspective(float(fov), float(aspect), float(znear), float(zfar))
#print "project"
#print project
#print "fov " + str(fov)
#print "aspect " + str(aspect)
#print "znear " + str(znear)
#print "zfar " + str(zfar)

frame = np.ones((int(height),int(width),3), np.uint8)
scaler = int(0.9*float(width/maxx-minx))
#print "scaler " + str(scaler)
#print "adderx" + str(adderx)
#print "addery" + str(addery)

p31 = []
p32 = []
p33 = []

line31 = []
line32 = []
line33 = []

line21 = []
line22 = []
line23 = []

for idx, vector in enumerate(readmesh.vectors):
#    if(readmesh.normals[idx][2] > 0):
#    print readmesh.normals[idx]
    p31 = Point3(float(readmesh.vectors[idx][0][0]),float(readmesh.vectors[idx][0][1]),float(readmesh.vectors[idx][0][2]))
    p32 = Point3(float(readmesh.vectors[idx][1][0]),float(readmesh.vectors[idx][1][1]),float(readmesh.vectors[idx][1][2]))
    p33 = Point3(float(readmesh.vectors[idx][2][0]),float(readmesh.vectors[idx][2][1]),float(readmesh.vectors[idx][2][2]))
    p21 = project * p31   
    p22 = project * p32   
    p23 = project * p33   
    p21 = Point2(float(p21.x/p21.z),float(p21.y/p21.z)) 
    p22 = Point2(float(p22.x/p22.z),float(p22.y/p22.z)) 
    p23 = Point2(float(p23.x/p23.z),float(p23.y/p23.z)) 
    line21 = LineSegment2(p21, p22)
    line22 = LineSegment2(p21, p23)
    line23 = LineSegment2(p22, p23)
    tp00 = (int(scaler*line21.p.x + adderx),                     int(scaler*line21.p.y + addery))
    tp01 = (int(scaler*line21.p.x + scaler*line21.v.x + adderx), int(scaler*line21.p.y + scaler*line21.v.y + addery))
    tp10 = (int(scaler*line22.p.x + adderx),                     int(scaler*line22.p.y + addery))
    tp11 = (int(scaler*line22.p.x + scaler*line22.v.x + adderx), int(scaler*line22.p.y + scaler*line22.v.y + addery))
    tp20 = (int(scaler*line23.p.x + adderx),                     int(scaler*line23.p.y + addery))
    tp21 = (int(scaler*line23.p.x + scaler*line23.v.x + adderx), int(scaler*line23.p.y + scaler*line23.v.y + addery))
    cv2.line(frame,tp00,tp01,(0,0,255), 1)   # draw line
    cv2.line(frame,tp10,tp11,(0,0,255), 1)   # draw line
    cv2.line(frame,tp20,tp21,(0,0,255), 1)   # draw line
cv2.imshow("window",frame)
cv2.waitKey(0)
