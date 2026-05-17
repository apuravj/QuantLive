from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def main():
    print("Training in process....")
    model.train(data='/Users/apuravjain/Desktop/Hackathon/SplitData/data.yaml', epochs = 300, imgsz=640, patience=25)
    print("Training completed....")

if __name__== '__main__':
    main()

