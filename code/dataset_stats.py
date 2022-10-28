
import glob
import matplotlib.pyplot as plt
import numpy as np

train_dir = 'data/pneumonia/chest_xray/train'
test_dir = 'data/pneumonia/chest_xray/test'

pneumonia_train_images = glob.glob(train_dir + "/PNEUMONIA/*")
normal_train_images = glob.glob(train_dir + "/NORMAL/*")

pneumonia_test_images = glob.glob(test_dir + "/PNEUMONIA/*")
normal_test_images = glob.glob(test_dir + "/NORMAL/*")

print(f'pneumonia train images = {len(pneumonia_train_images)}')
print(f'normal train images = {len(normal_train_images)}')

print(f'pneumonia test images = {len(pneumonia_test_images)}')
print(f'normal test images = {len(normal_test_images)}')

plt.figure(dpi=200)
plt.pie(x=np.array([len(pneumonia_train_images), len(normal_train_images)]), autopct="%.1f%%", explode=[0.2,0], labels=["pneumonia", "normal"], pctdistance=0.5)
plt.title("TRAINING DATA STATISTICS", fontsize=14)
plt.show()
