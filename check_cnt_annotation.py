import cv2
import numpy as np
import json
import os
import random
import shutil
DEBUG=1

dst="/home/asad/annotated_700_cucumber/removed"
file_extensions=[".png",".jpeg",".JPEG",".jpg",".JPG"]

def draw_bounds(img,data,color=(255,0,0)):
    color= [random.randint(25,255) for i in range(3)]
    ##color=[255,255,255]
    #color=255
    color=tuple(color)
    for point in data:
        cv2.circle(img,tuple(point),3,color)
    
    #cv2.imshow("Img",cv2.resize(img,(720,1280)))
    #cv2.waitKey(0)
    


class load_cucumbers:

    def __init__(self,data_path,numbers=None):
        self.data_path=data_path
        self.numbers=numbers
        self.data=[]
        self.skipped=0
        self.read_data()
        
    
    def read_data(self):
        index=0
        for file in (os.listdir(self.data_path)):
            if file.endswith(".json"):
                for ext in file_extensions:
                    imagefile=file.split(".")[0]+ext
                    if (os.path.exists(self.data_path+"/"+imagefile)):
                        break
                index+=1
                image=cv2.imread(self.data_path+imagefile,-1)
                if image is None:
                    self.skipped+=1
                    continue
                j_file=os.path.exists(self.data_path+"/"+file)
                with open(self.data_path+"/"+file) as f:
                    data = json.load(f)
                cucs=data["shapes"]
                if DEBUG:
                    all_cnts=[]
                    print(len(cucs))
                    for points in cucs:
                        draw_bounds(image,points)

                    cv2.imshow("Overlayed",cv2.resize(image,(720,1280)))
                    if cv2.waitKey(0) == ord('a'):
                        shutil.move(self.data_path+imagefile,dst)
                        shutil.move(self.data_path+"/"+file,dst)
                        

                dat={"img":image,"cucs":cucs}
                self.data.append(dat)
                if self.numbers is not None:
                    if (index==200):
                        kk=1
                    if (index>=self.numbers):
                        break
        print(self.skipped)
        print("Loaded all images")
                
                




if __name__ == "__main__":
    cuc=load_cucumbers("/home/asad/corrected_annotated/json/",700)
    cuc.read_data()
    #print(len(cuc))

