import numpy
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import json
from json import JSONEncoder

scale=1
debug=True
args=argparse.ArgumentParser("Parse data labels parsing")
args.add_argument("--data_dir",type=str,default="/home/asad/test_cuc")
opt=args.parse_args()

if opt.data_dir==None:
    print("Specify the data dir")
    exit()

def get_color():
    color1 = (list(np.random.choice(range(256), size=3)))  
    color =[int(color1[0]), int(color1[1]), int(color1[2])] 
    return color

def convert(o):
    if isinstance(o, np.int32): return int(o)  
    raise TypeError

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def check_json_annotation(json_path,img):
    with open(json_path, 'r') as outfile:
        data = json.load(outfile)
        k=data['shapes']
        for p in data['shapes']:
            p_ar=np.array(p['points'],dtype=np.int32)
            img=cv2.fillConvexPoly(img,p_ar,color=get_color())
            #img=cv2.fillPoly(img,p_ar,color=get_color())
    return img


for filename in os.listdir(opt.data_dir):
    if filename.endswith((".png.png",".json")):
        continue
    data_json={}
    data=Path(opt.data_dir)
    img_path=data.joinpath(Path(filename))
    json_path=data.joinpath(Path(filename[:-4]+".json"))
    ann_path=data.joinpath(Path(filename+".png"))
    if not (img_path.exists()):
        print(f"Image file {img_path} does not exists") 
    if not (ann_path.exists()):
        print(f"Annotation file {ann_path} does not exists") 
    img=cv2.imread(str(img_path))
    lbl=cv2.imread(str(ann_path),-1)
    sc_w,sc_h=img.shape[:-1]
    print(sc_w*sc_h)
    #c_w*=scale
    #sc_h*=scale
    #img=cv2.resize(img,(sc_w,sc_h))
    unq_labels=sorted(np.unique(lbl))
    vis_img=np.zeros_like(img,dtype=np.uint8)
    shapes=[]
    for l in unq_labels:
        if l==0:
            continue
        obj=np.where(lbl==l)
        blank_img=np.zeros_like(img,dtype=np.uint8)
        blank_img[obj]=(255,255,255)
        gray = cv2.cvtColor(blank_img, cv2.COLOR_BGR2GRAY) 
        # Find Canny edges 
        edged = cv2.Canny(gray, 20, 200) 
        contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_cont_size=0 
        max_contour=contours[0]
        for contour in contours:
            if (contour.size>max_cont_size):
                max_cont_size=contour.size
                max_contour=contour

        sh_con={"label": "1","points": max_contour.squeeze()}
        print(f"Number of cotours per mask {len(contours)}")
        shapes.append(sh_con)
        #cv2.imshow("blank image",blank_img)
        #cv2.waitKey(0)
        

    img_name=str(img_path).split("/")[-1]
    data_json["imagePath"]=img_name
    data_json["imageHeght"]=sc_h
    data_json["imageWidth"]=sc_w
    data_json["shapes"]=shapes
    
    with open(json_path, 'w') as outfile:
        json.dump(data_json, outfile,cls=NumpyArrayEncoder)

    vis_img=check_json_annotation(json_path,vis_img)

    vis_img=cv2.addWeighted(img,0.5,vis_img,0.5,1)
    cv2.imshow("Label",vis_img)
    cv2.waitKey(0)

    