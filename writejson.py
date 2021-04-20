import json
import cv2
import numpy as np
import json
import os
import random
import shutil




def save_json(json_name,image_name,img,data,dst_dir="/home/asad/annotated_700_cucumber/modified_cuc_data"):
    outfile={}
    outfile["img_name"]=image_name
    height,width=img.shape[:2]
    outfile["height"]=height
    outfile["width"]=width
    outfile["shapes"]=data
    with open(os.path.join(dst_dir,json_name), 'w') as out:
        json.dump(outfile, out)
    cv2.imwrite(os.path.join(dst_dir,image_name),img)


def draw_bounds(img,data,color=(255,0,0)):
    #color= [random.randint(25,255) for i in range(3)]
    ##color=[255,255,255]
    color=255
    #color=tuple(color)
    for point in data:
        cv2.circle(img,tuple(point),1,color)
    
    #cv2.imshow("KK",cv2.resize(img,(1280,720)))
    #cv2.waitKey(0)
    
    zero_img=np.zeros(img.shape[:2],dtype=np.uint8)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    max_cnt=None
    max_sze=0
    for cnt in contours:
        if len(cnt)>max_sze:
            max_sze=len(cnt)
            max_cnt=cnt
    if max_cnt is None:
        print("Contour is None")
        exit()
    cv2.drawContours(zero_img, max_cnt, -1, 255, 3)
    max_cnt=max_cnt.squeeze().tolist()
    return max_cnt
    #cv2.imshow("Input",img)
    #cv2.imshow("Contour",zero_img)
    #cv2.waitKey(0)

    
file_extensions=[".png",".jpeg",".JPEG",".jpg",".JPG"]
DEBUG=1
dst="/home/asad/test/json"

class load_cucumbers:

    def __init__(self,data_path,numbers=None):
        self.data_path=data_path
        self.numbers=numbers
        self.data=[]
        self.skipped=0
        self.read_data()
        
    
    def read_data(self):
        index=0
        all_labels=[file for file in os.listdir(self.data_path) if file.endswith((".jpg.png",".png.png"))]
        for file in (all_labels):
                for ext in file_extensions:
                    imagefile=file.split(".")[0]+ext
                    if (os.path.exists(self.data_path+"/"+imagefile)):
                        break
                index+=1
                image=cv2.imread(os.path.join(self.data_path,imagefile),-1)
                if image is None:
                    self.skipped+=1
                    continue
                print(f"Processing {imagefile}")
                l_file=os.path.join(self.data_path,file)
                j_file=os.path.join(self.data_path,file.split(".")[0]+".json")
                #with open(self.data_path+"/"+file) as f:
                #    data = json.load(f)
                label_img=cv2.imread(l_file,-1)
                all_labels=np.unique(label_img)
                all_cnts=[]
                for label in all_labels:
                    if label==0:
                        continue
                    [cc,rr]=np.where(label_img==label)
                    XY = list(zip(rr, cc))
                    blankimg=np.zeros(image.shape[:2],dtype=np.uint8)
                            #draw_bounds(image,points['points'])
                    cnt=draw_bounds(blankimg,XY)
                    all_cnts.append(cnt)
                save_json(j_file,imagefile,image,all_cnts)
                    #cv2.imshow("Overlayed",image)
                if cv2.waitKey(1) == ord('a'):
                        shutil.move(self.data_path+imagefile,dst)
                        shutil.move(self.data_path+"/"+file,dst)
                        
                print(index)
                #dat={"img":image,"cucs":cucs}
                #self.data.append(dat)
                if self.numbers is not None:
                    if (index==454):
                        kk=1
                    if (index>=self.numbers):
                        break
        print(self.skipped)
        print("Loaded all images")
                
                




if __name__ == "__main__":
    cuc=load_cucumbers("/media/asad/ADAS_CV/cuc/axel_april_evaluate_gt",1500)
    #cuc.read_data()
    #print(len(cuc))

