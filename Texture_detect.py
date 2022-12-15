import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imshow
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
import cv2

def entropy_mask_viz(image):
    image_gray = rgb2gray(image)
    entropy_image = entropy(image_gray, disk(6))
    scaled_entropy = entropy_image / entropy_image.max()
    f_size = 24
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    
    ax[0].set_title('Greater Than Threshold', 
                     fontsize = f_size)
    threshold = scaled_entropy > 0.5
    image_a = np.dstack([image[:,:,0]*threshold,
                            image[:,:,1]*threshold,
                            image[:,:,2]*threshold])
    ax[0].imshow(image_a)
    ax[0].axis('off')
    
    ax[1].set_title('Less Than Threshold', 
                     fontsize = f_size)
    threshold = scaled_entropy < 0.8
    image_b = np.dstack([image[:,:,0]*threshold,
                            image[:,:,1]*threshold,
                            image[:,:,2]*threshold])
    ax[1].imshow(image_b)
    ax[1].axis('off')
    fig.tight_layout()
    return [image_a, image_b]


# img = cv2.imread("BoatImage.png")
video = cv2.VideoCapture('videoBoat.mp4')
retval, image = video.read()
while(image.any()):
    retval, image = video.read()
    image_a, image_b = entropy_mask_viz(image)
    cv2.imshow("Gray", image_a)  
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
        
video.release()
cv2.destroyAllWindows()