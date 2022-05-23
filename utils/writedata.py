from .CreateXMLFile import *
# from .Detector import *
#
# def detect(model_path):
#     detector = FRCNNOD(model_path + 'frozen_inference_graph.pb', model_path + 'labelmap.pbtxt')

def writexml(img,objs, writepath,folder,filename):
    height, width, depth = img.shape
    CXMLF = CreateXMLFile()
    CXMLF.folder.text = folder
    CXMLF.filename.text = filename + '.jpg'
    CXMLF.path.text = writepath + '/' + filename + '.jpg'
    CXMLF.width.text = str(width)
    CXMLF.height.text = str(height)
    CXMLF.depth.text = str(depth)
    for i in range(len(objs)):
        if i == len(objs) - 1:
            endstyle = '\n'
        else:
            endstyle = '\n\t'
        CXMLF.appendObject(objs[i], endstyle)
    CXMLF.writeETree(writepath + '/' + filename + '.xml')

def writeRecordxml(img,objs, writepath,folder,filename, framenum):
    height, width, depth = img.shape
    CXMLF = CreateXMLFile()
    CXMLF.folder.text = folder
    CXMLF.filename.text = filename + '.jpg'
    CXMLF.path.text = writepath + '/' + filename + '.jpg'
    CXMLF.width.text = str(width)
    CXMLF.height.text = str(height)
    CXMLF.depth.text = str(depth)
    CXMLF.appendFrameNum(framenum)
    for i in range(len(objs)):
        if i == len(objs) - 1:
            endstyle = '\n'
        else:
            endstyle = '\n\t'
        CXMLF.appendObject(objs[i], endstyle)
    CXMLF.writeETree(writepath + '/' + filename + '.xml')
