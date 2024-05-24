#  Modified YOLOv7 

### Installation Procedure

#### Step 1

     conda create --name env python=3.7 -y && conda activate env

#### Step 2

     git clone https://github.com/pinkson311/My_thesis_bachelor.git

#### Step 3

     cd My_thesis_bachelor/REAL-TIME_Distance_Estimation_with_YOLOV7
     
#### Step 4     

     pip install -r requirements.txt
     
#### Step 5 

     wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
     
#### Step 6    

     python detect.py --save-txt --weights yolov7.pt --conf 0.4 --source 0 --model_dist model@1535470106.json --weights_dist model@1535470106.h5 




