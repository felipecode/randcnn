

from input_peatones_data  import DataSetManager
import time 

dataset = DataSetManager('../PeatonesDetectionTF/KITTI/training/image_2','../PeatonesDetectionTF/Dataset','../PeatonesDetectionTF/Dataset',[128,64,3],[224,224,3])

start_time = time.time()
images, labels =dataset.train.next_batch(200)
duration = time.time() - start_time
print labels
print images.shape
print duration