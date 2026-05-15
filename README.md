(Proyecto de ejemplo) Integración con TensorBoard

TensorBoard permite visualizar el progreso del entrenamiento (pérdida, precisión, etc.). Pasos rápidos:

1. Instalar dependencias (si aún no):

```
pip install torch torchvision tensorboard
```

2. Ejecutar el script de entrenamiento que registra en TensorBoard:

```
python src/train_with_tensorboard.py --epochs 10 --batch-size 32 --logdir runs/lenet_experiment
```

3. Iniciar el servidor de TensorBoard y abrir el navegador en http://localhost:6006:

```
tensorboard --logdir runs/lenet_experiment
```

Los logs por defecto se crean en `runs/lenet_experiment`.

