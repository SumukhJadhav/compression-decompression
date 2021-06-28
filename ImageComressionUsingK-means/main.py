import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from PIL import Image,ImageChops
from sklearn.utils import shuffle
from time import time
import math
import sys
from PyQt5 import QtCore, QtWidgets, uic,QtGui
import os
import re
from pathlib import Path



n_colors =64
class Ui(QtWidgets.QMainWindow,):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi("front.ui", self)
        
        self.upload.released.connect(lambda: uploads())


        def compress(filename):
            self.UpdataData_4.setText("Compressing")
            t1 = time()

            #filename = "xxa.jpg"
            img  = Image.open(filename)

            img = np.array(img, dtype=np.float64) / 255

            w, h, d = original_shape = tuple(img.shape)
            #assert d == 3
            image_array = np.reshape(img, (w * h, d))


            image_array_sample = shuffle(image_array, random_state=0)[:1000]
            kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

            print("Predicting colors...")
            t0 = time()
            labels = kmeans.predict(image_array)
            print("Predicted in %0.3fs." % (time() - t0))
        

            def recreate_image(codebook, labels, w, h):
                d = codebook.shape[1]
                image = np.zeros((w, h, d))
                label_idx = 0
                for i in range(w):
                    for j in range(h):
                        image[i][j] = codebook[labels[label_idx]]
                        label_idx += 1
                return image
            
            plt.figure(2)   
            plt.clf()
            plt.axis('off')
            plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
            figure = plt.gcf()
            figure.set_size_inches(16, 12)
            filename2 = Path(filename).stem
            plt.savefig(f'{filename2} Compressed.jpg',bbox_inches='tight',pad_inches = 0,dpi =100)   
            #plt.show()
            tt = time() - t1
            tt = float("{0:.2f}".format(tt))
            print("Compressed in %0.3fs." % (time() - t1))

            s1 = os.path.getsize(filename)
            s2 = os.path.getsize(f'{filename2} Compressed.jpg')

            Nsize = (s1 - s2) / math.pow(1024,2)
            Nsize = float("{0:.2f}".format(Nsize))
            print(Nsize)
            self.UpdataData_4.setText("Completed")
            self.UpdataData_3.setText("Your Image was compressed by " + str(Nsize) + "mb in " + str(tt) + "s")
        #compress('C:/Users/sumuk/OneDrive/Pictures/GIGL/1.png')
        def uploads():
            print("pressed")
            filename = str(QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', r"C:/"))
            filename =(re.sub('[()]', '', filename))
            filename = filename[1:- 16]
            print ('Path file :', filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                compress(filename)
            else:
                self.UpdataData_4.setText("Invalid file")

def main():
    
    app = QtWidgets.QApplication(sys.argv)

    window = Ui()
    window.show()

    app.exec_()

if __name__ == "__main__":
    main()
