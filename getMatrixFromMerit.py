from mevis import *
import os

def run(macro):
    fileName = "matrix.mat"
    workingDir = "E:/BachelorThesis/WorkingDirectory"
    
    currentNetwork = MLAB.IDE().currentDocument().network()
    for module in currentNetwork.modules():
        if module.type() == "MERIT":
            MeritMatrix = module.field("transformationMatrix").value
            if (MeritMatrix is not None):
                fileName = os.path.join(workingDir, fileName)
                print ("Writing transformation matrix to: " + fileName)
                f = open(fileName, "w")
                for i in range (0,len(MeritMatrix)):
                    for value in MeritMatrix[i]:
                        f.write (str(value))
                        f.write (" ")
                    f.write ("\n")
                f.close()        
            break
            
    