import cv2
import os

root = '/home/chirag/Desktop/2021-22 Sem1/COL780: Computer Vision/Project/LA-Transformer/data/'

for path, subdirs, files in os.walk(root):
    for path in subdirs:
    	if(path == 'val'):
    		continue
    	pt = os.path.join(root,path)
    	for path2, subdirs2, files2 in os.walk(pt):
    		for path2 in subdirs2:
    			pt_final = os.path.join(pt,path2)
    			for _,_,files in os.walk(pt_final):
		    		for file in files:
		    			img_pth = os.path.join(pt_final,file)
			    		img = cv2.imread(img_pth)
			    		img_flip_lr = cv2.flip(img, 1)
			    		img_path_final = img_pth[:-4]+"new.png"
			    		cv2.imwrite(img_path_final, img_flip_lr)
