from ultralytics import YOLO
from Utilities import Utilities

class YoloModel:
    DEFAULT_MODEL = "yolo11n.pt"
    
    def __init__(self):
        self.model = None
        
    def load_weights(self, weights_path=None):
        try:
            if weights_path is None:
                weights_path = self.DEFAULT_MODEL
            self.model = YOLO(weights_path)
            Utilities.show_message(f"Модель {weights_path} загружена успешно!")
            return True
        except Exception as e:
            Utilities.show_error_message(f"Ошибка загрузки весов по пути: {weights_path}")
            return False
        
    def detect(self, image):
        if self.model is None:
            self.load_weights()
            
        try:
            results = self.model(image)
            
            bboxes = []
            for result in results:
                for box in result.boxes:
                    bboxes.append(box.xywhn[0])
            
            return True, image, bboxes
            
        except Exception as e:
            Utilities.show_error_message(f"Ошибка обнаружения: {e}")
            return False, image, None