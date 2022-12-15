import math
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from utils import detect_horizon_line

def main():
    """Main logic"""

    vid = True
    if(vid):
        cap = cv2.VideoCapture('sampleBoat3.mp4') # File available in https://drive.google.com/file/d/1rYwFoNRrQPnEltrM8cpJ8f91Ow1iQDcI/view?usp=share_link
        counter = 3
        # cap = cv2.VideoCapture('rtsp://admin:@10.0.0.3:554')
        while(cap.isOpened()):
            ret, image = cap.read()
            if(image is not None):
                # Detecting mountain area ---- start
                color = ('b','g','r')
                for i,col in enumerate(color):
                    histr = cv2.calcHist([image],[i],None,[256],[0,256])
                    plt.plot(histr,color = col)
                    plt.xlim([0,256])
                plt.show()

                b = image.copy()
                # set green and red channels to 0
                b[:, :, 1] = 0
                b[:, :, 2] = 0


                g = image.copy()
                # set blue and red channels to 0
                g[:, :, 0] = 0
                g[:, :, 2] = 0

                r = image.copy()
                # set blue and green channels to 0
                r[:, :, 0] = 0
                r[:, :, 1] = 0
                
                # RGB - Blue
                cv2.imshow('B-RGB', b)

                # RGB - Green
                cv2.imshow('G-RGB', g)

                # RGB - Red
                cv2.imshow('R-RGB', r)
            

                lo=np.array([50,50,50])
                hi=np.array([100,75,85])

                # Mask image to only select browns
                mask=cv2.inRange(image,lo,hi)

                # Change image to red where we found brown
                image[mask>0]=(0,0,255)
                # Detecting mountain area ---- end

                # Detecting Waterline ------ start

                # Convert to graycsale
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Blur the image for better edge detection
                img_blur = cv2.GaussianBlur(img_gray, (5,5
                ), 0) 
                
                
                # Canny Edge Detection
                template = cv2.cvtColor(cv2.imread('edge_template3.jpg'), cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=120) # Canny Edge Detection
                edges = cv2.subtract(edges, template)
                # Copy edges to the images that will display the results in BGR
                cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                cdstP = np.copy(cdst)
                
                lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, 5, 0, 0)
                
                if lines is not None:
                    for i in range(0, len(lines)):
                        rho = lines[i][0][0]
                        theta = lines[i][0][1]
                        a = math.cos(theta)
                        b = math.sin(theta)
                        if(theta < 1.0):
                            print(theta)
                            continue
                        x0 = a * rho
                        y0 = b * rho
                        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                        cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
                
                
                linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
                
                if linesP is not None:
                    for i in range(0, len(linesP)):
                        l = linesP[i][0]
                        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
                
                cv2.imshow("Source", image)
                cv2.imshow('frame',edges)
                if cv2.waitKey(1) & 0xFF == ord('d'):
                    print('save image')
                    counter += 1
                    cv2.imwrite('edge_' + str(counter) + '.jpg', edges)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()