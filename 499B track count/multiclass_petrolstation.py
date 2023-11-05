import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone

model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('20231001124901335.avi')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
cy1=440


tracker1=Tracker()
tracker2=Tracker()
tracker3=Tracker()
tracker4=Tracker()



counter1=[]
counter2=[]
counter3=[]
counter4=[]
offset=6
while True:    
    ret,frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue
    
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list1=[]
    motorcycle=[]
    list2=[]
    car=[]
    list3=[]
    truck=[]
    list4=[]
    bus=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'motorcycle' in c:
            list1.append([x1,y1,x2,y2])
            motorcycle.append(c)
        elif 'car' in c:
            list2.append([x1,y1,x2,y2])
            car.append(c)
        elif 'truck' in c:
            list3.append([x1,y1,x2,y2])
            truck.append(c)
        elif 'bus' in c:
            list4.append([x1,y1,x2,y2])
            bus.append(c)    
        
    bbox1_idx=tracker1.update(list1)
    bbox2_idx=tracker2.update(list2)
    bbox3_idx=tracker3.update(list3)
    bbox4_idx=tracker4.update(list4)
##########################################BIKE##################################
    for bbox1 in bbox1_idx:
        for i in motorcycle:
            x3,y3,x4,y4,id1=bbox1
            cxm=int(x3+x4)//2
            cym=int(y3+y4)//2
            #if cym<(cy1+offset) and cym>(cy1-offset):
               #cv2.circle(frame,(cxm,cym),4,(0,255,0),-1)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),1)
            
            cvzone.putTextRect(frame, 'BIKE', (x3, y3), 1, 1)
            
            #cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
            
            if y3 < cy1 + offset and y4 > cy1 - offset and id1 not in counter1:
                counter1.append(id1)
                
           # if counter1.count(id1)==0:
               # counter1.append(id1)
 
 ###############################CAR#############################################
    for bbox2 in bbox2_idx:
        for h in car:
            x5,y5,x6,y6,id2=bbox2
            cxc=int(x5+x6)//2
            cyc=int(y5+y6)//2
             #if cym<(cy1+offset) and cym>(cy1-offset):
               #cv2.circle(frame,(cxc,cyc),4,(0,255,0),-1)
            cv2.rectangle(frame,(x5,y5),(x6,y6),(0,0,255),1)
            
            cvzone.putTextRect(frame, 'CAR', (x5, y5), 1, 1)
            
            #cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
            
            if y5 < cy1 + offset and y6 > cy1 - offset and id2 not in counter2:
                counter2.append(id2)
                
           # if counter1.count(id1)==0:
               # counter1.append(id1)
########################TRUCK###################################################
    for bbox3 in bbox3_idx:
        for t in truck:
            x7,y7,x8,y8,id3=bbox3
            cxt=int(x7+x8)//2
            cyt=int(y7+y8)//2
             #if cym<(cy1+offset) and cym>(cy1-offset):
               #cv2.circle(frame,(cxc,cyc),4,(0,255,0),-1)
            cv2.rectangle(frame,(x7,y7),(x8,y8),(0,0,255),1)
            
            cvzone.putTextRect(frame, 'TRUCK', (x5, y5), 1, 1)
            
            #cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
            
            if y7 < cy1 + offset and y8 > cy1 - offset and id3 not in counter3:
                counter3.append(id3)
                
           # if counter1.count(id1)==0:
               # counter1.append(id1)
############################BUS###############################################
    for bbox4 in bbox4_idx:
        for b in bus:
            x9,y9,x10,y10,id4=bbox4
            cxt=int(x9+x10)//2
            cyt=int(y9+y10)//2
             #if cym<(cy1+offset) and cym>(cy1-offset):
               #cv2.circle(frame,(cxc,cyc),4,(0,255,0),-1)
            cv2.rectangle(frame,(x9,y9),(x10,y10),(0,0,255),1)
            
            cvzone.putTextRect(frame, 'BUS', (x5, y5), 1, 1)
            
            #cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
            
            if y9 < cy1 + offset and y10 > cy1 - offset and id4 not in counter4:
                counter4.append(id4)
                
           # if counter1.count(id1)==0:
               # counter1.append(id1)           
               
    cv2.line(frame,(2,cy1),(794,cy1),(0,0,255),2)

  
    motorcyclec=(len(counter1))
    carc=(len(counter2))
    truckc=(len(counter3))
    busc=(len(counter3))
    
    cvzone.putTextRect(frame,f'motorcyclec:-{motorcyclec}',(19,30),2,1)
    cvzone.putTextRect(frame,f'carc:-{carc}',(18,71),2,1)
    cvzone.putTextRect(frame,f'truckc:-{truckc}',(18,116),2,1)
    cvzone.putTextRect(frame,f'busc:-{busc}',(18,162),2,1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()






