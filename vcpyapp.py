
from ultralytics import YOLO


with open('/content/coco128-mini.yaml', 'w') as f:
    f.write(yaml)


model = YOLO('yolov8n.pt')
model.train(data='/content/coco128-mini.yaml', epochs=3, imgsz=640, batch=4)

results = model('/content/coco128-mini/images/')
for r in results:
    r.show()
