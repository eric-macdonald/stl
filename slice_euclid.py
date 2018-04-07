import doctest
import stl
from stl import mesh
import math
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import sys
import cv2
from euclid import *

filename = sys.argv[1]
slices = sys.argv[2]
height = sys.argv[3]
width = sys.argv[4]

min_area = 0.0001
max_area = 100000000

def complexity(frame):
    edged = cv2.Canny(frame, 5, 300)
    (img, cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for idx, c in enumerate(cnts):
        if(float(cv2.contourArea(c)) > float(min_area)) and (float(cv2.contourArea(c)) < float(max_area)):
            print "hierarchy = " + str(hierarchy[0][idx])
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            frame = cv2.drawContours(frame, [c], 0, (255, 255, 255), 1)
    cv2.imshow('framed', frame)
    cv2.waitKey(0)
    # videof.write(frame)
    return len(cnts)
 
        

# checks to see if point C is in line and in between points A and B
def isBetween(a, b, c):
    V1 = (c - a)
    V2 = (b - a)
    Cross = V1.cross(V2)
    MagCross  = Cross.magnitude()
    if(MagCross > 0.00001):
       return False
    Dotn = V2.dot(V1)
    Doto = V2.dot(V2)
    first = (Dotn > Doto)
    second = (Doto < 0) and (Dotn > 0)
    third  = (Doto > 0) and (Dotn < 0)
    if(first or second or third):
        return False
    else:
        return True


def plot_mesh(obj):
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(obj.vectors))
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    scale = obj.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)
    pyplot.show()

def initiate_video(readmesh, height, width, outputfile):
    output_size = (int(width),int(height))
    codec = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 5 
    videof = cv2.VideoWriter()
    success = videof.open(outputfile,codec,fps,output_size,True) 
    return videof, success

def scale_mesh(readmesh, height, width):
    minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(readmesh)
    xwidth = maxx - minx
    ywidth = maxy - miny

    print xwidth
    print ywidth

    if(xwidth>ywidth):
        scaler = xwidth
    else:
        scaler = ywidth

    scaler = float(width)/(1.1*float(scaler))

    if(minx<0):
        adderx = abs(scaler*minx) + 5 
    else:
        adderx = 5 

    if(miny<0):
        addery = abs(scaler*miny) + 5 
    else:
        addery = 5 

    print "adderx " + str(adderx)
    print "addery " + str(addery)
    print "scaler " + str(scaler) 
    return adderx, addery, scaler

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
    print "minx " + str(minx)
    print "maxx " + str(maxx)
    print "miny " + str(miny)
    print "maxy " + str(maxy)
    print "minz " + str(minz)
    print "maxz " + str(maxz)
    return minx, maxx, miny, maxy, minz, maxz


def initiate_video(readmesh, height, width, outputfile):
    output_size = (int(width),int(height))
    codec = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 5 
    videof = cv2.VideoWriter()
    success = videof.open(outputfile,codec,fps,output_size,True) 
    return videof, success

#p11 and p12 are the triangle segment and p21 and p22 are the slice plane intersection
def find_line_intersection(p11, p12, p21, p22):
    V1 = (p12 - p11)  # direction of line segment of triangle
    V2 = (p22 - p21)  # direction of line of slice plane intersection 
    V3 = (p21 - p11) # line segment from one line to the other 
    V5 = V1.cross(V2)
    V6 = V3.cross(V2)
    parallel1 = V5.dot(V6)
    V5_mag = V5.magnitude()
    V6_mag = V6.magnitude()
    if(V5_mag > 0.000001):
        a =  V6_mag/V5_mag
        if(abs(a)>0.00001):
            if(parallel1 > 0.000001):
                piadd = a*(V1)
                pi = p11 + piadd
                if(isBetween(p11, p12, pi)):
                    return True, pi
                else: 
                    return False, pi
            else:
                piadd = a*(V1)
                pi = p11 - piadd
                if(isBetween(p11, p12, pi)):
                    return True, pi
                else: 
                    return False, pi
        else:
            return False, (0,0,0) 
    else:
        return False, (0,0,0) 


def slice(readmesh, slices, height, width, outputfile):
    total_complexity = 0
    frame = np.ones((int(height),int(width),3), np.uint8)
    minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(readmesh)
    adderx, addery, scaler = scale_mesh(readmesh, height, width)
    depths = np.arange(minz, (maxz + float(maxz-minz)/int(slices)), float(maxz-minz)/(int(slices)))
    videof, success = initiate_video(readmesh, height, width, outputfile)
    if(not success):
        print "failed to start video"
    for depth in depths:
        idx = 0
        frame = np.ones((int(height),int(width),3), np.uint8)
        print "depth ################# " + str(depth)
        for idx, vector in enumerate(readmesh.vectors):
            pt0 = Point3(float(readmesh.vectors[idx][0][0]),float(readmesh.vectors[idx][0][1]),float(readmesh.vectors[idx][0][2]))
            pt1 = Point3(float(readmesh.vectors[idx][1][0]),float(readmesh.vectors[idx][1][1]),float(readmesh.vectors[idx][1][2]))
            pt2 = Point3(float(readmesh.vectors[idx][2][0]),float(readmesh.vectors[idx][2][1]),float(readmesh.vectors[idx][2][2]))
            ps0 = Point3(0,0,depth)
            ps1 = Point3(0,1,depth)
            ps2 = Point3(1,0,depth)
            tline1 = LineSegment3(pt0, pt1)   
            tline2 = LineSegment3(pt1, pt2)   
            tline3 = LineSegment3(pt0, pt2)   
            sline1 = LineSegment3(ps0, ps1)   
            sline2 = LineSegment3(ps0, ps2)   
            tn = tline1.v.cross(tline2.v) 
            sn = sline1.v.cross(sline2.v)
            tn = tn.normalized()
            sn = sn.normalized()
            dt = pt1.dot(tn)
            ds = float(ps0.dot(sn))
            if((tn == sn) or (tn == -1*sn)):
                if (((dt == ds) and (tn == sn)) or ((dt == -1*ds) and (tn == -1*sn))):
                    tp00 = (int(scaler*tline1.p.x + adderx),                     int(scaler*tline1.p.y + addery))
                    tp01 = (int(scaler*tline1.p.x + scaler*tline1.v.x + adderx), int(scaler*tline1.p.y + scaler*tline1.v.y + addery))
                    tp10 = (int(scaler*tline2.p.x + adderx),                     int(scaler*tline2.p.y + addery))
                    tp11 = (int(scaler*tline2.p.x + scaler*tline2.v.x + adderx), int(scaler*tline2.p.y + scaler*tline2.v.y + addery))
                    tp20 = (int(scaler*tline3.p.x + adderx),                     int(scaler*tline3.p.y + addery))
                    tp21 = (int(scaler*tline3.p.x + scaler*tline3.v.x + adderx), int(scaler*tline3.p.y + scaler*tline3.v.y + addery))
                    cv2.line(frame,tp00,tp01,(0,0,255), 1)   # draw line
                    cv2.line(frame,tp10,tp11,(0,0,255), 1)   # draw line
                    cv2.line(frame,tp20,tp21,(0,0,255), 1)   # draw line
            else:
                line = []
                plane_t = Plane(tn,dt)
                plane_s = Plane(sn,ds)
                intersection = plane_t.intersect(plane_s)
                status, p1 = find_line_intersection(pt0, pt1, intersection.p, intersection.p + intersection.v) 
                if(status):
                    line.append(p1)
                status, p2 = find_line_intersection(pt1, pt2, intersection.p, intersection.p + intersection.v) 
                if(status):
                    line.append(p2)
                status, p3 = find_line_intersection(pt0, pt2, intersection.p, intersection.p + intersection.v) 
                if(status):
                    line.append(p3)
                if(len(line) == 3):
                    p1 = 0
                if(len(line) == 2):
                    p1 = (int(line[0][0]*scaler + adderx), int(line[0][1]*scaler + addery))
                    p2 = (int(line[1][0]*scaler + adderx), int(line[1][1]*scaler + addery))
                    cv2.line(frame,p1,p2,(255,0,0), 2)   # draw line
                elif(len(line) == 1):
                    p1 = (int(line[0][0]*scaler + adderx), int(line[0][1]*scaler + addery))
                    cv2.circle(frame, p1, 2, (0,255,0)) 
        videof.write(frame)
        cmetric1 = complexity(frame)
        print "complexity = " + str(cmetric1)
        total_complexity = total_complexity + cmetric1
    print "total complexity for this orientation = " +  str(total_complexity)
#        cv2.imshow("window",frame)
#        cv2.waitKey(0)
                
# Main program
filename = sys.argv[1]
slices = sys.argv[2]
height = sys.argv[3]
width = sys.argv[4]

print "reading mesh"
readmesh0 = mesh.Mesh.from_file(filename)

#print "pre rotate"
#readmesh0.rotate([1,0,0], math.pi/4)

#plot_mesh(readmesh0)
print "slicing mesh"
slice(readmesh0, slices, height, width, 'meshX00.avi')

print "rotating mesh"
readmesh0.rotate([0,1,0], math.pi/2)
print "slicing second time for mesh"
slice(readmesh0, slices, height, width, 'meshX90.avi')
