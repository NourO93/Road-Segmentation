import matplotlib.image as mpimg
import numpy as np

def read_image(image_filename):
    img = mpimg.imread(image_filename)
    a=np.asarray(img)
    (n,m)=a.shape
    return a
    
def mean_f_score(arr,ground_truth,tresh,patch_size=16):
    (n,m)=arr.shape
    fp,fn,tp,tn=0,0,0,0
    subm=np.zeros((n,m))
    patch=np.ones((patch_size,patch_size))
    for i in range(0,n,patch_size):
        for j in range(0,m,patch_size):
            if arr[i:i+patch_size,j:j+patch_size].mean()>tresh:
                subm[i:i+patch_size,j:j+patch_size]=patch
            else:
                subm[i:i+patch_size,j:j+patch_size]=0.0*patch
    for i in range(n):
        for j in range(m):
            if ground_truth[i][j]>0.5:
                if subm[i][j]>0.5: tp+=1
                else: fn+=1
            else:
                if subm[i][j]>0.5: fp+=1
                else: tn+=1
    p=tp*1.0/(tp+fp)
    r=tp*1.0/(tp+fn)
    return 2*p*r/(p+r)

    
def mfs_files(file_candidate,file_ground_truth,tresh=0.25):
    # takes the file and ground truth and outputs the score
    # tresh determines how is rounding done for patches of size 16x16
    # important: roads should be in white (correspond to zeros in the file)
    arr=1-read_image(file_candidate)
    ground_truth=1-read_image(file_ground_truth)
    return mean_f_score(arr,ground_truth,tresh)
    
