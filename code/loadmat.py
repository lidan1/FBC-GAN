import scipy.io
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import numpy as np
from copy import deepcopy

img_path = '/Users/lidan/Downloads/images/199.Winter_Wren/Winter_Wren_0035_1510546070.jpg'
mat_path = '/Users/lidan/Downloads/annotations/annotations-mat/199.Winter_Wren/Winter_Wren_0035_1510546070.mat'
img = Image.open(img_path).convert('RGB')
width, height = img.size
plt.figure()
plt.imshow(img)
plt.show()

mat = scipy.io.loadmat('/Users/lidan/Downloads/annotations/annotations-mat/199.Winter_Wren/Winter_Wren_0035_1510546070.mat')

print(mat)
seg = mat['seg']
bbx = mat['bbox'][0,0]

# plt.figure()
# plt.imshow(seg)
# plt.show()

print(seg)
print(bbx)
print(bbx['left'][0,0])
print(bbx['top'][0,0])
print(bbx['right'][0,0])
print(bbx['bottom'][0,0])
bbox = [bbx['left'][0,0],bbx['top'][0,0],bbx['right'][0,0]-bbx['left'][0,0],bbx['bottom'][0,0]-bbx['top'][0,0]]

r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
center_x = int((2 * bbox[0] + bbox[2]) / 2)
center_y = int((2 * bbox[1] + bbox[3]) / 2)
y1 = np.maximum(0, center_y - r)
y2 = np.minimum(height, center_y + r)
x1 = np.maximum(0, center_x - r)
x2 = np.minimum(width, center_x + r)
fimg = deepcopy(img)
fimg_arr = np.array(fimg)
fimg = Image.fromarray(fimg_arr)
print(x1,x2,y1,y2)
print(width, height)
cimg = img.crop([x1, y1, x2, y2])

plt.figure(2)
plt.imshow(cimg)
plt.show()


img_path = '/Users/lidan/Downloads/cub2011/segmentations/002.Laysan_Albatross/Laysan_Albatross_0002_1027.png'
img = Image.open(img_path)
width, height = img.size
print(img.size)
print(np.array(img))
print(np.array(img).shape)
plt.figure()
plt.imshow(img)
plt.show()

img_path = '/Users/lidan/Downloads/birds/images/002.Laysan_Albatross/Laysan_Albatross_0002_1027.png'

# thresholding
img = img.point(lambda p : p > 0)



log_dir = "./drive/MyDrive/zippin/finegan_lidan/output/birds_2021_08_10_12_54_12/Log"
# ./drive/MyDrive/zippin/finegan_lidan/code
# os.makedirs(logs_base_dir, exist_ok=True)
# tensorboard --logdir "/Volumes/GoogleDrive/MyDrive/zippin/finegan_lidan/output/bird

tensorboard --logdir /Volumes/GoogleDrive/My\ Drive/zippin/finegan_lidan/output/birds_2021_08_10_12_54_12/Log 