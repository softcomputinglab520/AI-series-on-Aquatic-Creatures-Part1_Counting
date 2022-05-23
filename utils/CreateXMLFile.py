# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:27:59 2018

@author: 123
"""
"""
If you only want to prepare object detection task.
Please set some parameters following:
1.<segmented>0</segmented> in annotation
2.<pose>Unspecified</pose> in object
3.<database>Unknown</database> in source.(You can replace Unknown by your dataset name)
4.if ((xmin<=1) or (xmax>=width-1) or (ymin<=1) or (ymax>=height-1)) truncated set 1 
  else truncated set 0 in object
5.<difficult>0</difficult> in object when confidence higher than 0.9(90%)

"""

import xml.etree.cElementTree as ET
from xml.etree import ElementTree
import numpy as np

class CreateXMLFile():
    
    def __init__(self):
        self.annotation = ET.Element('annotation')
        self.folder = ET.SubElement(self.annotation, 'folder')
        self.filename = ET.SubElement(self.annotation, 'filename')
        self.path = ET.SubElement(self.annotation, 'path')
        self.source = ET.SubElement(self.annotation, 'source')
        self.database = ET.SubElement(self.source, 'database')
        self.size = ET.SubElement(self.annotation, 'size')
        self.width = ET.SubElement(self.size, 'width')
        self.height = ET.SubElement(self.size, 'height')
        self.depth = ET.SubElement(self.size, 'depth')
        self.segmented = ET.SubElement(self.annotation, 'segmented')
        # self.folder.text=''
        # self.filename.text=''
        # self.path.text=''
        self.database.text = 'Unknown'
        # self.width.text=''
        # self.height.text=''
        # self.depth.text=''
        self.segmented.text = '0'
        self.annotation.text = '\n\t'
        self.folder.tail = '\n\t'
        self.filename.tail = '\n\t'
        self.path.tail = '\n\t'
        self.source.text = '\n\t\t'
        self.database.tail = '\n\t'
        self.source.tail = '\n\t'
        self.size.text = '\n\t\t'
        self.size.tail = '\n\t'
        self.width.tail = '\n\t\t'
        self.height.tail = '\n\t\t'
        self.depth.tail = '\n\t'
        self.segmented.tail = '\n\t'
    
    def appendObject(self, list1, endstyle):
        obj = ET.SubElement(self.annotation, 'object')
        obj.text = '\n\t\t'
        obj.tail = endstyle
        name = ET.SubElement(obj, 'name')
        name.tail = '\n\t\t'
        pose = ET.SubElement(obj, 'pose')
        pose.tail = '\n\t\t'
        truncated = ET.SubElement(obj, 'truncated')
        truncated.tail = '\n\t\t'
        difficult = ET.SubElement(obj, 'difficult')
        difficult.tail = '\n\t\t'
        bndbox = ET.SubElement(obj, 'bndbox')
        bndbox.text = '\n\t\t\t'
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.tail = '\n\t\t\t'
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.tail = '\n\t\t\t'
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.tail = '\n\t\t\t'
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.tail = '\n\t\t'
        bndbox.tail = '\n\t'
        name.text = list1[0]
        pose.text = list1[1]
        truncated.text = str(list1[2])
        difficult.text = str(list1[3])
        xmin.text = str(list1[4][0])
        ymin.text = str(list1[4][1])
        xmax.text = str(list1[4][2])
        ymax.text = str(list1[4][3])

    def appendFrameNum(self, num):
        framenum = ET.SubElement(self.annotation, 'framenum')
        framenum.text = str(num)
        framenum.tail = '\n\t'

    def writeETree(self, writepath):
        tree = ET.ElementTree(self.annotation)
        tree.write(writepath)
