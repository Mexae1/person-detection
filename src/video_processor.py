"""Модуль для обработки видеофайлов с YOLO26."""

import cv2
import os
import time
from typing import Optional, Dict, List
from pathlib import Path
from .detector import PersonDetector


class VideoProcessor:
    """
    Класс для обработки видео с детекцией людей через YOLO26.
    
    Attributes:
        detector: Экземпляр PersonDetector
        input_path: Путь к входному видео
        output_path: Путь к выходному видео
    """
    
    def __init__(self, 
                 detector: PersonDetector, 
                 input_path: str, 
                 output_path: str):
        """
        Инициализация процессора видео.
        
        Args:
            detector: Инициализированный детектор YOLO26
            input_path: Путь к входному видео
            output_path: Путь для сохранения результата
        """
        self.detector = detector
        self.input_path = input_path
        self.output_path = output_path
        
        # Проверка существования входного файла
        if not os.path.exists(input_path):
            raise FileNotFoundError(
                f"Входной видеофайл не найден: {input_path}"
            )
        
        # Создание директории для выходного файла
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_video(self, 
                     progress_callback: Optional[callable] = None,
                     show_fps: bool = True) -> Dict:
        """
        Обработка видео с детекцией людей.
        
        Args:
            progress_callback: Функция для отслеживания прогресса
            show_fps: Отображать FPS на выходном видео
            
        Returns:
            Словарь со статистикой обработки
        """
        # Открываем входное видео
        cap = cv2.VideoCapture(self.input_path)
        
        if not cap.isOpened():
            raise ValueError(
                f"Не удалось открыть видеофайл: {self.input_path}"
            )
        
        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n=== Параметры видео ===")
        print(f"Разрешение: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Всего кадров: {total_frames}")
        print(f"Длительность: {total_frames/fps:.2f} сек\n")
        
        # Создаем writer для выходного видео
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            fps, 
            (width, height)
        )
        
        if not out.isOpened():
            raise ValueError(
                f"Не удалось создать выходной файл: {self.output_path}"
            )
        
        # Статистика
        frame_count = 0
        person_count_stats: List[int] = []
        processing_times: List[float] = []
        start_time = time.time()
        
        print("Начало обработки...\n")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Засекаем время обработки кадра
            frame_start = time.time()
            
            # Детекция людей
            detections = self.detector.detect_persons(frame)
            person_count_stats.append(len(detections))
            
            # Отрисовка детекций
            output_frame = self.detector.draw_detections(frame, detections)
            
            # Добавляем FPS на кадр
            if show_fps and len(processing_times) > 0:
                avg_fps = 1.0 / (sum(processing_times[-30:]) / 
                                len(processing_times[-30:]))
                fps_text = f"FPS: {avg_fps:.1f} | Persons: {len(detections)}"
                cv2.putText(
                    output_frame, 
                    fps_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            
            # Запись обработанного кадра
            out.write(output_frame)
            
            # Время обработки кадра
            frame_time = time.time() - frame_start
            processing_times.append(frame_time)
            
            frame_count += 1
            
            # Callback для прогресса
            if progress_callback and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                progress_callback(
                    progress, 
                    frame_count, 
                    total_frames, 
                    current_fps
                )
        
        # Освобождаем ресурсы
        cap.release()
        out.release()
        
        # Финальная статистика
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        avg_persons = (sum(person_count_stats) / len(person_count_stats) 
                      if person_count_stats else 0)
        max_persons = max(person_count_stats) if person_count_stats else 0
        min_persons = min(person_count_stats) if person_count_stats else 0
        
        stats = {
            'total_frames': frame_count,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'avg_persons': avg_persons,
            'max_persons': max_persons,
            'min_persons': min_persons,
            'video_resolution': f"{width}x{height}",
            'output_path': self.output_path
        }
        
        self._print_final_stats(stats)
        
        return stats
    
    def _print_final_stats(self, stats: Dict):
        """Вывод финальной статистики обработки."""
        print(f"\n{'='*50}")
        print("ОБРАБОТКА ЗАВЕРШЕНА")
        print(f"{'='*50}")
        print(f"Обработано кадров: {stats['total_frames']}")
        print(f"Время обработки: {stats['total_time']:.2f} сек")
        print(f"Средний FPS: {stats['avg_fps']:.2f}")
        print(f"Разрешение: {stats['video_resolution']}")
        print(f"\nСтатистика детекций:")
        print(f"  Среднее кол-во людей: {stats['avg_persons']:.2f}")
        print(f"  Минимум: {stats['min_persons']}")
        print(f"  Максимум: {stats['max_persons']}")
        print(f"\nРезультат сохранен: {stats['output_path']}")
        print(f"{'='*50}\n")
