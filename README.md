Шаги по улучшению:
1. Для улучшения качества детекции можно использовать Faster R-CNN, RT‑DETR или LW‑DETR.
2. Можно изменить трэш-холд чтобы избавиться от неточных детекций.
3. Добавить поддержку GPU для быстрой обработки видео.
4. Дообучить на crowd датасетах

Запуск:
1. Клонируйте репозиторий
      git clone https://github.com/mexae1/person-detection.git
2. Перейдите в директорию проекта
    cd person-detection
3. python -m venv venv
    Windows активация
    venv\Scripts\activate
    Linux/macOS активация
    source venv/bin/activate
4. Скачивание библиотек
    pip install -r requirements.txt
6. Запуск
    python -m src.main --input data/crowd.mp4 --output output/result.mp4

Параметры:

Параметр	Описание	Тип	Значение по умолчанию	Обязательный
1. --input	Путь к входному видеофайлу	str		Да
2. --output	Путь для сохранения результата	str		Да
3. --model	Модель YOLO26 для использования	str	yolo26x.pt	Нет
4. --conf	Порог уверенности детекции (0.0-1.0)	float	0.30	Нет
5. --iou	Порог IoU для фильтрации (0.0-1.0)	float	0.45	Нет
6. --device	Устройство для инференса	str	auto	Нет
