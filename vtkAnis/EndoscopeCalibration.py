__rcs_info__ = {
    #
    #  Creation Information
    #
    'module_name':'$RCSfile: EndoscopeCalibration.py,v $',
    'creator':'cwedlake <cwedlake@imaging.robarts.ca>',
    'project':'Atamai Surgical Planning',
    #
    #  Current Information
    #
    'author':'$Author: cwedlake $',
    'version':'$Revision: 1.3 $',
    'date':'$Date: 2008/09/25 20:20:10 $',
    }
try:
    __version__ = __rcs_info__['version'].split(' ')[1]
except:
    __version__ = '0.0'



"""
EndoscopeCalibration - Used to calibrate the a tracked endoscope

  Will calibrate one endoscope based on the callback set for the item.
  
"""

import os
import tkFileDialog, tkMessageBox, vtk, Pmw
import tkApp
import sys
import string
import time
import vasst
import math
from SimpleActors import *

from tkAppModule import *
import Tkinter

from vtkAnisPython import *

# make boolean types if not defined
try:
    (True, False)
except NameError:
    True = 1
    False = 0


__rcs_info__ = {
    #
    #  Creation Information
    #
    'module_name':'$RCSfile: EndoscopeCalibration.py,v $',
    'creator':'cwedlake <cwedlake@imaging.robarts.ca>',
    'project':'Atamai Surgical Planning',
    #
    #  Current Information
    #
    'author':'$Author: cwedlake $',
    'version':'$Revision: 1.3 $',
    'date':'$Date: 2008/09/25 20:20:10 $',
    }
try:
    __version__ = __rcs_info__['version'].split(' ')[1]
except:
    __version__ = '0.0'


"""
generic module setup
"""
class EndoscopeCalibration(tkAppModule):    
    """generic module with 5 buttons:
                 anything with 'stub' in the name should be replaced!!!
    """
###################################################################
    def __init__(self, guiframe, name=None):
        tkAppModule.__init__(self, guiframe=guiframe, name="Endoscope Calibration")
        
        vasst.GetTrackedTools().AddCallback('EndoCalEndoscope',self._SetEndoscope)
        vasst.GetTrackedTools().AddCallback('EndoCalLeftEndoscope',self._SetLeftEndoscope)
        vasst.GetTrackedTools().AddCallback('EndoCalRightEndoscope',self._SetRightEndoscope)
        vasst.GetTrackedTools().AddCallback('EndoCalPointer',self._SetPointer)

        if (os.path.exists(".\\EndoscopeCalibrationImages") == 0): # creates the necessary directory
            os.mkdir(".\\EndoscopeCalibrationImages")              # if it doesn't already exist

        self._DefaultDirectory = ".\\EndoscopeCalibrationImages\\"

        self._DefaultDirectory 
        self._Divots = []
        self.CurrentDivot=0

        self.Corners = []

        self._GridUp = 5
        self._GridLeft = 7
        self.__GridBoxSize = 5 #mmS

        self.__RightCalibrationMethod = vtkVideoCalibration()
        self.__LeftCalibrationMethod = vtkVideoCalibration()
        
        self._PointerTool = None
        self._EndoscopeTool = None
        self.Tracker = None
        self._LeftEndoscopeTool = None
        self.LeftTracker = None
        self._RightEndoscopeTool = None
        self.RightTracker = None

        self.__TotalImages = 0
        self.__TotalLeftImages = 0
        self.__TotalRightImages = 0
        
        self._sa = SimpleActors()
        self.ActorList = []   
        self.CameraList = []
        self.TransList = []
        self.toolCal = []

        self._sa.SetConeSource(5,15)

        self.ShiftHidden = 0
        self.ImageHidden =0
        self.ToolHidden =0
        self._EndoscopeTrackedInstrumentFactory = None
        self._LeftEndoscopeTrackedInstrumentFactory = None
        self._RightEndoscopeTrackedInstrumentFactory = None

    def BuildInterface(self):
        """Build menu interface
        """
        tkAppModule.BuildInterface(self)
        buttonWidth = 3
        buttonsFrame = Frame(self.GuiFrame,
                                    background=tkApp.GetBackgroundColor())

        divotButton = tkApp.Button(buttonsFrame,
                                           command=self.AddDivotPoint,
                                           text='Set Divot',
                                           width=buttonWidth*3)
        fdivotButton = tkApp.Button(buttonsFrame,
                                           command=self.ForceDivotPoint,
                                           text='F Divot',
                                           width=buttonWidth*3)
        sdivotButton = tkApp.Button(buttonsFrame,
                                           command=self.SaveDivotPoint,
                                           text='S Divot',
                                           width=buttonWidth*3)
        aquireButton = tkApp.Button(buttonsFrame,
                                           command=self.AquireImage,
                                           text='Aquire Image',
                                           width=buttonWidth*3)
        calibrateButton = tkApp.Button(buttonsFrame,
                                           command=self.CalibrateEndoscope,
                                           text='Calibrate',
                                           width=buttonWidth*3)
        saveButton = tkApp.Button(buttonsFrame,
                                           command=self.SaveImages,
                                           text='Save',
                                           width=buttonWidth*3)
        loadButton = tkApp.Button(buttonsFrame,
                                           command=self.LoadImages,
                                           text='Load',
                                           width=buttonWidth*3)         
        clearconesButton = tkApp.Button(buttonsFrame,
                                           command=self.ClearCones,
                                           text='Clear Cones',
                                           width=buttonWidth*6)

        HideToolButton = tkApp.Button(buttonsFrame,
                                           command=self.HideTools,
                                           text='Trans',
                                           width=buttonWidth*2)
        HideImageButton = tkApp.Button(buttonsFrame,
                                           command=self.HideImages,
                                           text='Img',
                                           width=buttonWidth*2)
        HideShiftButton = tkApp.Button(buttonsFrame,
                                           command=self.HideShifts,
                                           text='Calib',
                                           width=buttonWidth*2)
        GetStatsButton = tkApp.Button(buttonsFrame,
                                           command=self.GetStats,
                                           text='Get Stats',
                                           width=buttonWidth*2)
        LineButton = tkApp.Button(buttonsFrame,
                                           command=self.DrawLine,
                                           text='Line',
                                           width=buttonWidth*2)
        GetImagePairButton = tkApp.Button(buttonsFrame,
                                           command=self.GetImagePair,
                                           text='Acquire Stereo',
                                           width=buttonWidth*4)
        SaveStereoImagesButton = tkApp.Button(buttonsFrame,
                                           command=self.SaveImagesStereo,
                                           text='Save Stereo',
                                           width=buttonWidth*4)
        loadLeftButton = tkApp.Button(buttonsFrame,
                                           command=self.LoadLeftImages,
                                           text='Load Left',
                                           width=buttonWidth*3)
        loadRightButton = tkApp.Button(buttonsFrame,
                                           command=self.LoadRightImages,
                                           text='Load Right',
                                           width=buttonWidth*3) 
        LeftCalibrateButton = tkApp.Button(buttonsFrame,
                                           command=self.CalibrateLeftEndoscope,
                                           text='Left Calibrate',
                                           width=buttonWidth*4)
        RightCalibrateButton = tkApp.Button(buttonsFrame,
                                           command=self.CalibrateRightEndoscope,
                                           text='Right Calibrate',
                                           width=buttonWidth*4)

##         FocalScale = tkApp.Scale(buttonsFrame,
##                                      label="Scale",
##                                      width= buttonWidth*3,
##                                      from_=1,to=1000,
##                                      resolution=1,
##                                      command=self.__SetScaleValue )


        buttonsFrame.grid(row=0, column=0)
        divotButton.grid(row=0, column=0, columnspan=1)
        fdivotButton.grid(row=0, column=1, columnspan=1)
        sdivotButton.grid(row=0, column=2, columnspan=1)
        aquireButton.grid(row=1, column=0, columnspan=2)
        saveButton.grid(row=2, column=0, columnspan=1)
        loadButton.grid(row=2, column=1, columnspan=1)
        calibrateButton.grid(row=3, column=0, columnspan=2)
        clearconesButton.grid(row=4, column=0, columnspan=2)
        HideToolButton.grid(row=5, column=0, columnspan=1)
        HideImageButton.grid(row=5, column=1, columnspan=1)
        HideShiftButton.grid(row=5, column=2, columnspan=1)
        GetStatsButton.grid(row=6, column=0, columnspan=1)
        LineButton.grid(row=6, column=1, columnspan=1)
        GetImagePairButton.grid(row=7, column=0, columnspan=2)
        SaveStereoImagesButton.grid(row=8, column=0, columnspan=2)
        loadLeftButton.grid(row=9, column=0, columnspan=1)
        loadRightButton.grid(row=9, column=1, columnspan=1)
        LeftCalibrateButton.grid(row=10, column=0, columnspan=2)
        RightCalibrateButton.grid(row=11, column=0, columnspan=2)
##         FocalScale.grid(row=6, column=0, columnspan=3)
        
    def _SetEndoscope(self, instrument):
        """this method is passed to the TrackedTools class: when an actorfactory is
           atached to this callback, we will get a pointer to the TrackedInstrumentFactory
           holding the tool, and thus get access to the tool
        """
        self._EndoscopeTrackedInstrumentFactory = instrument
        self._EndoscopeTool = instrument.GetTrackerTool()
        self.Tracker = instrument.GetTrackerTool().GetTracker()
        video = instrument.GetVideoSource()
        self.Video = video
        self.__CalibrationMethod.singleCamera.SetTrackerTool(self._EndoscopeTool)
        self.__CalibrationMethod.singleCamera.SetVideoSource(video)
        self.__CalibrationMethod.singleCamera.SetImageWidth(video.GetFrameSize()[0])
        self.__CalibrationMethod.singleCamera.SetImageHeight(video.GetFrameSize()[1])

    def _SetLeftEndoscope(self, instrument):
        self._LeftEndoscopeTrackedInstrumentFactory = instrument
        self._LeftEndoscopeTool = instrument.GetTrackerTool()
        self.LeftTracker = instrument.GetTrackerTool().GetTracker()
        LVideo = instrument.GetVideoSource()
        self.LeftVideo = LVideo
        self.__CalibrationMethod.leftCamera.SetTrackerTool(self._LeftEndoscopeTool)
        self.__CalibrationMethod.leftCamera.SetVideoSource(LVideo)
        self.__CalibrationMethod.leftCamera.SetImageWidth(LVideo.GetFrameSize()[0])
        self.__CalibrationMethod.leftCamera.SetImageHeight(LVideo.GetFrameSize()[1])

    def _SetRightEndoscope(self, instrument):
        self._RightEndoscopeTrackedInstrumentFactory = instrument
        self._RightEndoscopeTool = instrument.GetTrackerTool()
        self.RightTracker = instrument.GetTrackerTool().GetTracker()
        RVideo = instrument.GetVideoSource()
        self.RightVideo = RVideo
        self.__CalibrationMethod.rightCamera.SetTrackerTool(self._RightEndoscopeTool)
        self.__CalibrationMethod.rightCamera.SetVideoSource(RVideo)
        self.__CalibrationMethod.rightCamera.SetImageWidth(RVideo.GetFrameSize()[0])
        self.__CalibrationMethod.rightCamera.SetImageHeight(RVideo.GetFrameSize()[1])

    def _SetPointer(self, instrument):
        """this method is passed to the TrackedTools class: when an actorfactory is
           atached to this callback, we will get a pointer to the TrackedInstrumentFactory
           holding the tool, and thus get access to the tool
        """
        self._PointerTrackedInstrumentFactory = instrument
        self._PointerTool = instrument.GetTrackerTool()
 
    def ShowInterface(self):
        tkAppModule.ShowInterface(self)

    def HideInterface(self):
        tkAppModule.HideInterface(self)

    def AddDivotPoint(self):
        if (self.CurrentDivot > 2):
            return
        
        tool = self._PointerTool
        total = 0.0
        x = 0.0
        y = 0.0
        z = 0.0
        if (tool != None):
            for i in xrange(50):
                tool.GetTracker().Update()
                if not (tool.IsOutOfView()):
                    position = tool.GetTransform().GetPosition()
                    print position
                    x += position[0]
                    y += position[1]
                    z += position[2]
                    total += 1
                    time.sleep(0.05)
            if total <= 10:
                print "Not enough points found"
                return
            self._Divots.append( [(x/total), (y/total), (z/total)] )            
            self.CurrentDivot += 1
            if self.CurrentDivot > 2:
                self.__CreateCheckboard()
                self.__CalibrationMethod.SetCheckerboard(self._Divots[0][0],self._Divots[0][1],self._Divots[0][2],
                                                         self._Divots[1][0],self._Divots[1][1],self._Divots[1][2],
                                                         self._Divots[2][0],self._Divots[2][1],self._Divots[2][2])
                
        self._sa.Sphere( [(x/total), (y/total), (z/total)], [], tkApp.GetPane("3D"),[0,1,0])

    def ForceDivotPoint(self):
        self._Divots = [[], [], []]

        self._Divots[0] = [11.029780558357288, -16.503514348106332, -82.279112946196861]
        self._Divots[1] = [11.172964443606515, 13.567252686255381, -82.369415033176722]
        self._Divots[2] = [11.172964443606515, 13.567252686255381, -82.369415033176722]
        
        self.CurrentDivot = 3
        self.__CreateCheckboard()
        self.__CalibrationMethod.SetCheckerboard(self._Divots[0][0],self._Divots[0][1],self._Divots[0][2],
                                                 self._Divots[1][0],self._Divots[1][1],self._Divots[1][2],
                                                 self._Divots[2][0],self._Divots[2][1],self._Divots[2][2])

        self._sa.Sphere( self._Divots[0], [], tkApp.GetPane("3D"),[0,0,1])
        self._sa.Sphere( self._Divots[1], [], tkApp.GetPane("3D"),[0,0,1])
        self._sa.Sphere( self._Divots[2], [], tkApp.GetPane("3D"),[0,0,1])



        tcb = vtk.vtkTransform()
        tcb.GetMatrix().DeepCopy( self.__CalibrationMethod.GetToCheckerboard() )
        self._sa.Cone(tcb,  self.ActorList, tkApp.GetPane("3D"),[1, 1,1])

        
        fcb = vtk.vtkTransform()
        fcb.GetMatrix().DeepCopy( self.__CalibrationMethod.GetFromCheckerboard())
        self._sa.Cone(fcb,  self.ActorList, tkApp.GetPane("3D"),[.5, 0.5,0.5])

        temp = vtk.vtkTransform()
        vtk.vtkMatrix4x4().Multiply4x4(fcb.GetMatrix(),tcb.GetMatrix(),  temp.GetMatrix())
        self._sa.Cone(temp, self.ActorList, tkApp.GetPane("3D"),[.5, 1,0.5])
        
    def SaveDivotPoint(self):
        print self._Divots[0],"\n",self._Divots[1],"\n",self._Divots[2]
        
    def AquireImage(self):
        if self.CurrentDivot > 2:
            if (self._EndoscopeTool == None):
                print "Endoscope tool must be set or nothing productive will happen"
                return
            writer = vtk.vtkBMPWriter()
            writer.SetInput(0,self.Video.GetOutput())
            writer.SetFileName("E:/image%d.bmp" % self.__TotalImages)
            writer.Write()
            
            self.__CalibrationMethod.singleCamera.acquireImage()
            
            self.__TotalImages +=1
        else:
            print "Must have all the divots first"
        

    def CalibrateEndoscope(self):
        if self.CurrentDivot > 2 and self.__TotalImages >= 2:
            self.__CalibrationMethod.singleCamera.DoCalibration(1)

            value = self.__CalibrationMethod.singleCamera.GetNumberOfImages()
            print self.__CalibrationMethod.singleCamera.GetCalibration()
            self.__CalibrationMethod.singleCamera.PrintIntrinsic()

            if  self._EndoscopeTrackedInstrumentFactory != None:
                self._EndoscopeTrackedInstrumentFactory.ApplyDistortion(self.__CalibrationMethod.singleCamera)
            for i in xrange(value):
                trans = vtk.vtkTransform()
                camera = vtk.vtkTransform()
                toolCal = vtk.vtkTransform()
                
                camera.SetMatrix(self.__CalibrationMethod.singleCamera.GetImageMatrix(i))
                trans.SetMatrix(self.__CalibrationMethod.singleCamera.GetToolTransform(i).GetMatrix())

                vtk.vtkMatrix4x4().Multiply4x4( trans.GetMatrix(), self.__CalibrationMethod.singleCamera.GetCalibration() ,toolCal.GetMatrix())
                
                self._sa.Cone(camera, self.CameraList, tkApp.GetPane("3D"),[0, 1,0])
                self._sa.Cone(trans,  self.TransList, tkApp.GetPane("3D"),[0, 0,1], )
                self._sa.Cone(toolCal,  self.toolCal, tkApp.GetPane("3D"),[1, 1,0])
        
        else:
            print "Must have more than 2 images and divots set"

    def CalibrateLeftEndoscope(self):
        if self.CurrentDivot > 2 and self.__TotalLeftImages >= 2:
            self.__CalibrationMethod.leftCamera.DoCalibration(1)
            
            value = self.__CalibrationMethod.leftCamera.GetNumberOfImages()
            print self.__CalibrationMethod.leftCamera.GetCalibration()
            self.__CalibrationMethod.leftCamera.PrintIntrinsic()
            
            if  self._EndoscopeTrackedInstrumentFactory != None:
                self._EndoscopeTrackedInstrumentFactory.ApplyDistortion(self.__CalibrationMethod.leftCamera)
            for i in xrange(value):
                trans = vtk.vtkTransform()
                camera = vtk.vtkTransform()
                toolCal = vtk.vtkTransform()
                
                camera.SetMatrix(self.__CalibrationMethod.leftCamera.GetImageMatrix(i))
                trans.SetMatrix(self.__CalibrationMethod.leftCamera.GetToolTransform(i).GetMatrix())

                vtk.vtkMatrix4x4().Multiply4x4( trans.GetMatrix(), self.__CalibrationMethod.leftCamera.GetCalibration() ,toolCal.GetMatrix())
                
                self._sa.Cone(camera, self.CameraList, tkApp.GetPane("3D"),[0, 1,0])
                self._sa.Cone(trans,  self.TransList, tkApp.GetPane("3D"),[0, 0,1], )
                self._sa.Cone(toolCal,  self.toolCal, tkApp.GetPane("3D"),[1, 1,0])
        
        else:
            print "Must have more than 2 images and divots set"

    def CalibrateRightEndoscope(self):
        if self.CurrentDivot > 2 and self.__TotalRightImages >= 2:
            self.__CalibrationMethod.rightCamera.DoCalibration(1)
            
            value = self.__CalibrationMethod.rightCamera.GetNumberOfImages()
            print self.__CalibrationMethod.rightCamera.GetCalibration()
            self.__CalibrationMethod.rightCamera.PrintIntrinsic()
            
            if  self._EndoscopeTrackedInstrumentFactory != None:
                self._EndoscopeTrackedInstrumentFactory.ApplyDistortion(self.__CalibrationMethod.rightCamera)
            for i in xrange(value):
                trans = vtk.vtkTransform()
                camera = vtk.vtkTransform()
                toolCal = vtk.vtkTransform()
                
                camera.SetMatrix(self.__CalibrationMethod.rightCamera.GetImageMatrix(i))
                trans.SetMatrix(self.__CalibrationMethod.rightCamera.GetToolTransform(i).GetMatrix())

                vtk.vtkMatrix4x4().Multiply4x4( trans.GetMatrix(), self.__CalibrationMethod.rightCamera.GetCalibration() ,toolCal.GetMatrix())
                
                self._sa.Cone(camera, self.CameraList, tkApp.GetPane("3D"),[0, 1,0])
                self._sa.Cone(trans,  self.TransList, tkApp.GetPane("3D"),[0, 0,1], )
                self._sa.Cone(toolCal,  self.toolCal, tkApp.GetPane("3D"),[1, 1,0])
        
        else:
            print "Must have more than 2 images and divots set"
            
    def __CreateCheckboard(self):

        uDir = [0.0, 0.0, 0.0]
        vDir = [0.0, 0.0, 0.0]
        uLen = 0
        vLen = 0

        for i in xrange(3):
            uDir[i] = self._Divots[1][i] - self._Divots[0][i]
            uLen += uDir[i] * uDir[i]

            vDir[i] = self._Divots[2][i] - self._Divots[0][i]
            vLen += vDir[i] * vDir[i]

        uLen = math.sqrt(uLen)
        vLen = math.sqrt(vLen)
        print uLen
        print vLen
        for i in xrange(3):
            uDir[i] /= uLen
            vDir[i] /= vLen

        for v in range(self._GridUp):
            self.Corners.append([])
            for u in range(self._GridLeft):
                addr = (v*self._GridLeft)+u
                self.Corners[v].append([])
                self.Corners[v][u] = [ self._Divots[0][0] + (self.__GridBoxSize * u * uDir[0]) + (self.__GridBoxSize * v * vDir[0]),
                                       self._Divots[0][1] + (self.__GridBoxSize * u * uDir[1]) + (self.__GridBoxSize * v * vDir[1]),
                                       self._Divots[0][2] + (self.__GridBoxSize * u * uDir[2]) + (self.__GridBoxSize * v * vDir[2]),
                                       1.0 ]
 
    def SaveImages(self):
        Folder = self._DefaultDirectory+"\\"+time.strftime('%Y.%m.%d-%Hh',time.localtime(time.time())) 
        
        if (os.path.exists(Folder) == 0): # creates the necessary directory
            os.mkdir(Folder)             # if it doesn't already exist
        
        directory = tkFileDialog.askdirectory(initialdir = Folder, title="select directory", mustexist=1 )
        if (directory):
            self.__CalibrationMethod.SaveImages(directory)

    def SaveImagesStereo(self):
        Folder = self._DefaultDirectory+"\\"+time.strftime('%Y.%m.%d-%Hh',time.localtime(time.time())) 
        
        if (os.path.exists(Folder) == 0): # creates the necessary directory
            os.mkdir(Folder)             # if it doesn't already exist

        SubLeft = Folder+"/Left"
        SubRight = Folder+"/Right"

        if (os.path.exists(SubLeft) == 0):
            os.mkdir(SubLeft)

        if (os.path.exists(SubRight) == 0):
            os.mkdir(SubRight)

        #directory = tkFileDialog.askdirectory(initialdir = Folder, title="select directory", mustexist=1 )
        directoryL = tkFileDialog.askdirectory(initialdir = SubLeft, title="select directory", mustexist=1 )
        directoryR = tkFileDialog.askdirectory(initialdir = SubRight, title="select directory", mustexist=1 )
        if (directoryL and directoryR):
            self.__CalibrationMethod.SaveImagesStereo(directoryL,directoryR)
            
    def LoadImages(self):

        directory = tkFileDialog.askdirectory(initialdir = self._DefaultDirectory, title="select directory", mustexist=1 )
        if (directory):
            listing = os.listdir(directory)
            for file in listing:
                if (file[:5]!= "endoImage" and file[-4:] == ".bmp"):
                    imagePath = directory+"/"+file

                    transformPath = directory+"/"+file[:-4]+".transform"
                    self.__CalibrationMethod.LoadImage(imagePath, transformPath)
                    self.__TotalImages+=1

        return 0

    def LoadLeftImages(self):

        directory = tkFileDialog.askdirectory(initialdir = self._DefaultDirectory, title="select directory", mustexist=1 )
        if (directory):
            listing = os.listdir(directory)
            for file in listing:
                if (file[:5]!= "endoImageLeft" and file[-4:] == ".bmp"):
                    imagePath = directory+"/"+file

                    transformPath = directory+"/"+file[:-4]+".transform"
                    self.__CalibrationMethod.LoadLeftImage(imagePath, transformPath)
                    self.__TotalLeftImages+=1

        return 0

    def LoadRightImages(self):

        directory = tkFileDialog.askdirectory(initialdir = self._DefaultDirectory, title="select directory", mustexist=1 )
        if (directory):
            listing = os.listdir(directory)
            for file in listing:
                if (file[:5]!= "endoImageRight" and file[-4:] == ".bmp"):
                    imagePath = directory+"/"+file

                    transformPath = directory+"/"+file[:-4]+".transform"
                    self.__CalibrationMethod.LoadRightImage(imagePath, transformPath)
                    self.__TotalRightImages+=1

        return 0

    def ClearCones(self):
        for list in self.ActorList:
            for item in list:
                tkApp.GetPane("3D").GetRenderer().RemoveActor(item)
        
        for list in self.CameraList:
            for item in list:
                tkApp.GetPane("3D").GetRenderer().RemoveActor(item)
                
        for list in self.TransList:
            for item in list:
                tkApp.GetPane("3D").GetRenderer().RemoveActor(item)

        for list in self.toolCal:
            for item in list:
                tkApp.GetPane("3D").GetRenderer().RemoveActor(item)

        self.ActorList = []      
        self.CameraList = []
        self.TransList = []
        self.toolCal = []

    def HideTools(self):
        for list in self.TransList:
            for item in list:
                if self.ToolHidden:
                    item.GetProperty().SetOpacity(1)
                else:
                    item.GetProperty().SetOpacity(0)
        self.ToolHidden = not self.ToolHidden

    def HideImages(self):
        for list in self.CameraList:
            for item in list:
                if self.ImageHidden:
                    item.GetProperty().SetOpacity(1)
                else:
                    item.GetProperty().SetOpacity(0)
        self.ImageHidden = not self.ImageHidden

    def HideShifts(self):
        for list in self.toolCal:
            for item in list:
                if self.ShiftHidden:
                    item.GetProperty().SetOpacity(1)
                else:
                    item.GetProperty().SetOpacity(0)
        self.ShiftHidden = not self.ShiftHidden

    def GetStats(self):
        self.__CalibrationMethod.GetStats(self._Divots[0][0],self._Divots[0][1],self._Divots[0][2],
                                                 self._Divots[1][0],self._Divots[1][1],self._Divots[1][2],
                                                 self._Divots[2][0],self._Divots[2][1],self._Divots[2][2])

    def DrawLine(self):
        self.__CalibrationMethod.PrintLine()
        
    def GetImagePair(self):
        writer = vtk.vtkBMPWriter()
        writer.SetInput(0,self.LeftVideo.GetOutput())
        writer.SetFileName("E:/LeftImage%d.bmp" % self.__TotalLeftImages)
        writer.Write()

        self.__CalibrationMethod.leftCamera.AcquireImage()

        self.__TotalLeftImages +=1

        writer.SetInput(0,self.RightVideo.GetOutput())
        writer.SetFileName("E:/RightImage%d.bmp" % self.__TotalRightImages)
        writer.Write()

        self.__TotalRightImages +=1
               
        self.__CalibrationMethod.rightCamera.AcquireImage()

        
##     def __SetScaleValue(self,value):
##         self._EndoscopeTrackedInstrumentFactory.FocalDistance = int(value)
 
