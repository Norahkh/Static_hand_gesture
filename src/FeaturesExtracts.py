import numpy as np
import cv2
import mediapipe as mp
import math
#...........................


class HandFetuers:
    
    #Method Get Vector From 2 Points
    @classmethod
    def ConvertPoints2Vector(cls,p1,p2):
        vector=p2-p1
        vector[1]=-vector[1]
        return vector
    #...........................

    #Method Get Lenghte Vector
    @classmethod
    def GetLengthVector(cls,vector):
        return math.sqrt(np.dot(vector,vector))
    #...........................

    #Method Get Coner Between 2 Vectors
    @classmethod
    def GetConerBetween2Vector(cls,v1,v2):
        return  math.pi-math.acos((np.dot(v1,v2)/(cls.GetLengthVector(v1)*cls.GetLengthVector(v2))))
    #...........................

    #Method  Get Vectors Fingers Hand 16x2  4 for finger Thume and  3 echo fingers
    @classmethod 
    def GetVectorsFingers(cls,Points):
        FingersVectors=np.zeros((16,2))
        i,j=0,0
        while i<21:
            FingersVectors[j]=cls.ConvertPoints2Vector(Points[i],Points[i+1])
            j=j+1
            i=i+1+int((i+1)%4==0)
        return FingersVectors     
    #...........................

    # Method Get Conner For Aall Fingers Vectors Retuern 16x1.  4 coners for Finger Thumes and 3 Coners echo fingers 
    @classmethod
    def GetConnersFingers(cls,FingersVectors,Points):
        ConersFingersVectors=np.zeros((16,1))
        i,f=0,1
        while i<3:
            ConersFingersVectors[i]=cls.GetConerBetween2Vector(FingersVectors[i],FingersVectors[i+1])
            i=i+1
            
            
        ConersFingersVectors[i]=cls.GetConerBetween2Vector(
                                                          cls.ConvertPoints2Vector(Points[1],Points[2]),
                                                          cls.ConvertPoints2Vector(Points[2],Points[4])
                                                          )
        i=i+1
        while i<16:
            ConersFingersVectors[i]=cls.GetConerBetween2Vector(FingersVectors[i],FingersVectors[i+1])
            if (i+1)%3==0:
                i=i+1
                ConersFingersVectors[i]=cls.GetConerBetween2Vector(
                                                            cls.ConvertPoints2Vector(Points[i-f],Points[i-f+1]),
                                                            cls.ConvertPoints2Vector(Points[i-f+1],Points[i-f+3])
                                                                  )
                f=f-1
            i=i+1   
        return  ConersFingersVectors   
    #...........................


    # Method Get Hand is Flipe Or NoFlipe return 1 for Flipe Or 0
    @classmethod 
    def CheckFlipeHand(cls,points,isHandLeft):
        A,B=points[17,:2]-points[0,:2],points[5,:2]-points[0,:2]
        B[1],A[1]=-B[1],-A[1]
        Bo=((math.pi*2)+math.atan2(B[1], B[0]))%(math.pi*2)
        Ao=((math.pi*2)+math.atan2(A[1], A[0]))%(math.pi*2)
        
        if (Ao<=math.pi/2) and (Bo>=math.radians(270)):
            Bo=-Bo
        elif (Bo<=math.pi/2) and (Ao>=math.radians(270)):
            Ao=-Ao
        lable=0
        if (Bo<Ao)^isHandLeft:
            lable=1        
        return lable 
    #...........................

    #Method Get Dirction Hand  Conner return Value Between 0 to 2*Pi
    @classmethod
    def GetHandDirction(cls,points):
        A=points[9,:2]-points[0,:2]
        A[1]=-A[1]
        theat=((2*math.pi)+math.atan2(A[1],A[0]))%(2*math.pi)
        return theat
    #...........................

    #Method Get Conner Between Thumes and All Landmark Fingers return 4x9 
    @classmethod
    def GetConnersBetweenThumesAndFingers(cls,points):
        p0,p5,p17=points[0,:2],points[5,:2],points[17,:2]
        alf1,alf2=1,2
        mp0p5=np.array([[(alf1*p5[0]+p0[0])/(1+alf1),(alf1*p5[1]+p0[1])/(1+alf1)]]).reshape(p0.shape)
        cm=np.array([[(alf2*mp0p5[0]+p17[0])/(1+alf2),(alf2*mp0p5[1]+p17[1])/(1+alf2)]]).reshape(p0.shape)
        res=np.zeros((4,9))
        j=6
        for i in range(4):
            f=0   
            for n in range(2,5,1):   
                for z in range(3):
                    v=cls.ConvertPoints2Vector(cm, points[j+z,:2])
                    vcmp4=cls.ConvertPoints2Vector(points[n,:2],cm)
                    res[i,f]=cls.GetConerBetween2Vector(v,vcmp4)
                    f=f+1
            j=j+1
        return res
    #...........................

    #Method Get All Feteuers return 1x54
    @classmethod
    def GetFeteuers(cls,landmrks,ishandleft):    
        points=np.zeros((21,2))
        i=0
        for pos in  landmrks:  
            points[i]=[pos.x,pos.y]
            i=i+1
        FingersVectors=cls.GetVectorsFingers(points)
        ConersFingersVectors=cls.GetConnersFingers(FingersVectors,Points=points)
        IsFlipeHand=cls.CheckFlipeHand(points,isHandLeft=ishandleft)
        DirctionHand=cls.GetHandDirction(points)
        ConersThumeAndAllFingers=cls.GetConnersBetweenThumesAndFingers(points)
         
        Fetuers=np.concatenate([np.array([[int(IsFlipeHand),DirctionHand]]),ConersFingersVectors.T],axis=1)
        Fetuers=np.concatenate([Fetuers,ConersThumeAndAllFingers.reshape((1,36))],axis=1)
        return Fetuers
    #...........................
#......................................................End Class..............................  
        