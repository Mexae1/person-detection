"""Точка входа в программу детекции людей с YOLO26."""

"""Точка входа в программу детекции людей с YOLO26."""

import argparse
import os
import sys
from pathlib import Path

from .detector import PersonDetector
from .video_processor import VideoProcessor

def progress_callback(progress: float, 
                      current: int, 
                      total: int, 
                      fps: float):
    """
    Отображение прогресса обработки видео.
    
    Args:
        progress: Процент выполнения
        current: Текущий кадр
        total: Всего кадров
        fps: Текущий FPS обработки
    """
    bar_length = 40
    filled = int(bar_length * progress / 100)
    bar = ' ' * filled + ' ' * (bar_length - filled)
    
    print(f'\r[{bar}] {progress:.1f}% | '
          f'Кадр {current}/{total} | '
          f'FPS: {fps:.1f}', 
          end='', flush=True)


def main():
    """Основная функция программы."""
    parser = argparse.ArgumentParser(
        description='Детекция людей на видео с использованием YOLO26 '
                    '(самая последняя и мощная модель)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Путь к входному видеофайлу'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help='Путь для сохранения обработанного видео'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='yolo26x.pt',
        choices=['yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt', 
                'yolo26l.pt', 'yolo26x.pt'],
        help='Модель YOLO26 (n=fastest, x=most accurate)'
    )
    
    parser.add_argument(
        '--conf', 
        type=float, 
        default=0.30,
        help='Порог уверенности детекции (0.0-1.0)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='Порог IoU для фильтрации (0.0-1.0)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu', None],
        help='Устройство для инференса (auto если не указано)'
    )
    
    parser.add_argument(
        '--show-fps',
        action='store_true',
        default=True,
        help='Отображать FPS на выходном видео'
    )
    
    args = parser.parse_args()
    
    # Проверка существования входного файла
    if not os.path.exists(args.input):
        print(f"❌ ОШИБКА: Входной файл не найден: {args.input}")
        sys.exit(1)
    
    # Проверка расширения файла
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    if not any(args.input.lower().endswith(ext) 
              for ext in valid_extensions):
        print(f"⚠️ ВНИМАНИЕ: Неподдерживаемое расширение файла. "
              f"Рекомендуемые: {', '.join(valid_extensions)}")
    
    print("=" * 60)
    print("ДЕТЕКЦИЯ ЛЮДЕЙ С YOLO26 (JANUARY 2026)")
    print("=" * 60)
    print(f"Входной файл: {args.input}")
    print(f"Выходной файл: {args.output}")
    print(f"Модель: {args.model}")
    print(f"Порог уверенности: {args.conf}")
    print(f"Порог IoU: {args.iou}")
    print("=" * 60 + "\n")
    
    try:
        # Инициализация детектора с YOLO26
        print("🚀 Инициализация YOLO26 детектора...")
        detector = PersonDetector(
            model_name=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device
        )
        
        # Вывод информации о модели
        model_info = detector.get_model_info()
        print(f"\n✓ Модель загружена: {model_info['model_name']}")
        print(f"✓ Устройство: {model_info['device'].upper()}")
        print(f"✓ Задача: {model_info['task']}\n")
        
        # Инициализация процессора видео
        processor = VideoProcessor(
            detector, 
            args.input, 
            args.output
        )
        
        # Обработка видео
        stats = processor.process_video(
            progress_callback=progress_callback,
            show_fps=args.show_fps
        )
        
        print("\n✅ Обработка успешно завершена!")
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
