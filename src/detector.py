"""Модуль для детекции людей с использованием YOLO26."""

from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Optional
import torch


class PersonDetector:
    """
    Класс для детекции людей с использованием YOLO26.
    
    Attributes:
        model: Загруженная модель YOLO26
        conf_threshold: Порог уверенности для детекции
        iou_threshold: Порог IoU (для версий с NMS)
        person_class_id: ID класса 'person' в COCO dataset
        device: Устройство для инференса (cuda/cpu)
    """
    
    def __init__(self, 
                 model_name: str = 'yolo26x.pt',
                 conf_threshold: float = 0.30,
                 iou_threshold: float = 0.45,
                 device: Optional[str] = None):
        """
        Инициализация детектора с YOLO26.
        
        Args:
            model_name: Имя модели YOLO26 (n/s/m/l/x)
            conf_threshold: Минимальная уверенность для детекции
            iou_threshold: Порог IoU для фильтрации
            device: Устройство ('cuda', 'cpu', или None для автовыбора)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Загрузка модели {model_name} на {self.device}...")
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_class_id = 0
        
        print(f"Модель загружена успешно. Использование: {self.device.upper()}")
        
    def detect_persons(self, frame: np.ndarray) -> List[dict]:
        """
        Детекция людей на кадре с использованием YOLO26.
        
        Args:
            frame: Входной кадр в формате BGR
            
        Returns:
            Список словарей с информацией о детекциях
        """
        results = self.model.predict(
            frame, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.person_class_id],
            verbose=False,
            device=self.device
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
                
            for box in boxes:
                # Извлечение координат bbox
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Проверка валидности bbox
                if x2 > x1 and y2 > y1:
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'class_name': 'person',
                        'class_id': class_id
                    })
                
        return detections
    
    def draw_detections(self, 
                       frame: np.ndarray, 
                       detections: List[dict],
                       line_thickness: int = 2,
                       font_scale: float = 0.6) -> np.ndarray:
        """
        Отрисовка детекций на кадре с оптимизированной визуализацией.
        
        Args:
            frame: Исходный кадр
            detections: Список детекций
            line_thickness: Толщина линий bbox
            font_scale: Размер шрифта
            
        Returns:
            Кадр с отрисованными детекциями
        """
        output_frame = frame.copy()
        
        # Цветовая схема
        bbox_color = (0, 255, 0)  
        text_bg_color = (0, 255, 0)  
        text_color = (0, 0, 0) 
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Рисуем bounding box
            cv2.rectangle(
                output_frame, 
                (x1, y1), 
                (x2, y2), 
                bbox_color, 
                line_thickness
            )
            
            # Подготовка текста с метками
            label = f"{class_name}: {confidence:.2f}"
            
            # Получаем размер текста
            (text_width, text_height), baseline = cv2.getTextSize(
                label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                line_thickness
            )
            
            # Позиция текста (сверху bbox или снизу, если места нет)
            text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
            
            # Фон для текста (полупрозрачный)
            cv2.rectangle(
                output_frame, 
                (x1, text_y - text_height - baseline),
                (x1 + text_width + 4, text_y + baseline),
                text_bg_color, 
                -1
            )
            
            # Отрисовка текста
            cv2.putText(
                output_frame, 
                label, 
                (x1 + 2, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                text_color, 
                line_thickness,
                cv2.LINE_AA
            )
            
        return output_frame
    
    def get_model_info(self) -> dict:
        """
        Получение информации о модели.
        
        Returns:
            Словарь с параметрами модели
        """
        return {
            'model_name': self.model.model_name,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'task': self.model.task
        }
