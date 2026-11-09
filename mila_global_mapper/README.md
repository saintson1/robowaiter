# Global Mapper

Нода `dual_rgbd_global_mapper` строит глобальную двумерную карту по depth-кадрам двух камер и визуальной одометрии.

## Как работает

1. Получает depth и `CameraInfo` камер A/B.
2. Получает позу робота из `/dual_odometry/odom`.
3. Переводит точки глубины в `odom` и обновляет OccupancyGrid.
4. Публикует `/map` и `/dual_rgbd_global_mapper/status`.
5. Сохраняет `global_map.pgm` и `global_map.yaml`.

## Сервисы

- `/dual_rgbd_global_mapper/save` — сохранить текущую карту.
- `/dual_rgbd_global_mapper/reset` — очистить карту.
- `/dual_rgbd_global_mapper/finish_loop` — скорректировать ручное замыкание маршрута и сохранить карту.

## Запуск

```bash
bash run_global_mapping.sh
```

Для ИК-одометрии с raw depth:

```bash
bash run_global_ir_mapping.sh
```

