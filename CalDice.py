import nibabel as nib
import numpy as np
import os
from os import listdir

#label = nib.load('/home/yilinliu/ISMRM_Dataset/FullStructureLabels/subject207_RIGHT_all_labels_8bit_path_RightLeftAmygdalaNOSUBFIELDS.nii')
#output = nib.load('/home/yilinliu/Niftynet/models/deepmedic/segmentation_output/207segmentation_output/subject207_niftynet_out.nii.gz')

#convert to numpy array
#outputdata = output.get_data()
#labeldata = label.get_data()
#std_shape = label.shape
#outputdata_rs = np.reshape(outputdata,std_shape)

# --- Dice Score ---
def computeDice(outputdata,labeldata,n_classes=11):
    DiceArray = []
    
    for c_i in xrange(0,n_classes):
        idx_output = np.where(outputdata_rs.flatten() == c_i)[0]
        idx_label = np.where(labeldata.flatten() == c_i)[0]
#       print('idx_label.size ',idx_label.size)

        outArray = np.zeros(outputdata.size,dtype=np.bool)
        outArray[idx_output] = 1

        labelArray = np.zeros(labeldata.size, dtype=np.bool)
        labelArray[idx_label] = 1

        dsc = dice(outArray,labelArray)

        DiceArray.append(dsc)
    return DiceArray


def dice(im1, im2):
    """
    Computes the Dice coefficient
    ----------
    im1 : boolean array
    im2 : boolean array
    
    If they are not boolean, they will be converted.
    
    -------
    It returns the Dice coefficient as a float on the range [0,1].
        1: Perfect overlapping 
        0: Not overlapping 
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.size != im2.size:
       #print('img1.size ',im1.size)
        #print('img2.size ',im2.size)
        raise ValueError("Size mismatch between input arrays!!!")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return 1.0

    # Compute Dice 
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum
    
#Compute dice score
#label_path = '/home/yilinliu/ISMRM_Dataset/FullStructureLabels'
#output_path = '/home/yilinliu/Niftynet/models/deepmedic/segmentation_output'

#labels,outputs = listdir(label_path), listdir(output_path)
label = nib.load('/home/yilinliu/ISMRM_Dataset/test/labels/subject206_Sublabels.nii')
output = nib.load('/home/yilinliu/Niftynet/models/deepmedic/SubSegmentation_output/subject206_niftynet_out.nii.gz')
#for file in listdir(output_path):   
 #   if file.endswith(".csv"):
  #     os.remove(os.path.join(output_path,file))
   #    print('--------------------------------------------- inferred.csv deleted!!!')

#for file in listdir(label_path):
 #   filename = file.split("_")[0]
  #  newname = filename + '.nii'
   # os.rename(os.path.join(label_path,file),newname)

#for labelname,outputname in zip(labels,outputs):
 #   label, output = nib.load(os.path.join(label_path,labelname)), nib.load(os.path.join(output_path,outputname))
    #convert to numpy array
outputdata = output.get_data()
labeldata = label.get_data()
std_shape = label.shape
outputdata_rs = np.reshape(outputdata,std_shape)
#print('label ',labelname)
#print('output ',outputname)
print('Overall dice score: ', dice(labeldata,outputdata_rs))
DiceArray = computeDice(np.array(outputdata_rs),np.array(labeldata))
print('DiceArray length ', np.array(DiceArray).shape)
for i in DiceArray:
    print('Dice socre of class ', i)

#print('Dice score of class 0(background): ',DiceArray[0])
#print('Dice score of class 1(left Amyg): ',DiceArray[1])
#print('Dice score of class 2(right Amyg): ',DiceArray[2])
