# Live Odometry

Нода `dual_odometry` вычисляет положение робота по синхронизированным кадрам двух RealSense D435i.

## Как работает

1. Синхронизирует изображения и depth обеих камер.
2. Находит и отслеживает визуальные признаки между кадрами.
3. Вычисляет движение через PnP RANSAC и проверяет его по depth.
4. Объединяет оценки камер в позу корпуса.
5. Публикует одометрию и TF `odom → base_link`.

## Выходы

- `/dual_odometry/odom` — поза и скорость.
- `/dual_odometry/path` — пройденный путь.
- `/dual_odometry/status` — состояние трекинга.
- `/tf` — `odom → base_link`.

## Режимы запуска

RGB + depth:

```bash
bash run_live_odometry.sh
```

IR1 + depth:

```bash
bash run_live_ir_odometry.sh
```

