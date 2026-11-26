# Súper-resolución de imágenes de gatos con SRGAN

Red generativa adversarial (SRGAN) en **TensorFlow/Keras** que duplica la resolución de imágenes de gatos, entrenada con un dataset personalizado.

Proyecto del curso **Redes Neuronales y Deep Learning** (Miniproyecto 3: Redes generativas) — Universidad Autónoma de Occidente, Cali, Colombia.

<!-- Reemplaza por una imagen real: entrada | bicúbica | SRGAN -->
![Ejemplo de resultado](docs/ejemplo_resultado.png)

## Descripción

La súper-resolución toma una imagen de baja resolución (LR) y reconstruye una versión de mayor resolución (HR). Aquí se usa una SRGAN: un **generador** crea la imagen ampliada ×2 y un **discriminador** intenta distinguirla de una imagen real.

El dataset solo contiene imágenes de 64×64, sin versiones de mayor resolución. Por eso se usa un esquema **autosupervisado**: cada imagen es el objetivo (HR) y la entrada (LR, de 32×32) se genera reduciéndola ×2 con interpolación bicúbica. Una vez entrenada, la red se puede aplicar a las imágenes originales de 64×64 para obtener versiones de 128×128.

## Estructura del repositorio

```
.
├── FinalFinal_gatos.ipynb   # Notebook completo (datos, modelo, entrenamiento, validación, demo)
├── README.md
└── docs/
    └── ejemplo_resultado.png
```

## Dataset

- Imágenes de gatos de 64×64 píxeles, divididas en `train` y `validation`.
- **[COMPLETAR: número de imágenes por conjunto, fuente y proceso de recolección]**
- El dataset y los pesos entrenados no se incluyen en el repositorio. Estructura esperada en Google Drive:

```
MyDrive/
└── Dataset_Cats/
    ├── train/        # imágenes (se admiten subcarpetas)
    └── validation/
```

## Arquitectura

**Generador**
- Convolución 9×9 (64 filtros)
- 16 bloques residuales (Conv 3×3 → BatchNorm → PReLU → Conv 3×3 → BatchNorm + conexión de salto)
- Convolución + BatchNorm con conexión de salto global
- Bloque de submuestreo (Conv + PixelShuffle) que duplica el tamaño espacial
- Convolución 9×9 con activación `tanh`

**Discriminador**
- 8 bloques convolucionales (Conv → BatchNorm → LeakyReLU 0.2), de 64 a 512 filtros
- Dense(1024) → LeakyReLU → Dense(1, sigmoid)

**Pérdidas**

```
L_G = 1.0 · L_perceptual + 1e-4 · L_adversarial
```

- *Perceptual:* MSE entre características de VGG19 (`block3_conv3`, ImageNet, congelada).
- *Adversarial:* entropía cruzada binaria.
- *Discriminador:* entropía cruzada binaria con suavizado de etiquetas (real = 0.9).

## Entrenamiento

| Parámetro | Valor |
|---|---|
| Entrada → salida | 32×32 → 64×64 (×2) |
| Optimizador | Adam (β₁ = 0.9) |
| Tasa de aprendizaje | G: 1e-4 · D: 5e-5 |
| Batch | 8 |
| Épocas | 100 |
| Aumento de datos | Volteos aleatorios sincronizados |
| Selección del modelo | Mejor PSNR en validación |

Se guardan checkpoints periódicos y logs de TensorBoard en Google Drive, y el entrenamiento se puede reanudar si Colab se desconecta (ajustando `START_EPOCH`).

## Cómo ejecutarlo

1. Sube el dataset a Google Drive con la estructura indicada arriba.
2. Abre `FinalFinal_gatos.ipynb` en [Google Colab](https://colab.research.google.com/) y activa la GPU (*Entorno de ejecución → Cambiar tipo de entorno → GPU*).
3. Ejecuta las celdas en orden. La primera de rutas confirma que ve tus carpetas y cuántas imágenes hay.
4. Al final, la celda de **prueba de imagen propia** te permite subir una imagen, escalarla ×2 y descargar el resultado.

### Dependencias

Colab ya trae casi todo. Para ejecutarlo localmente necesitarías:

```
tensorflow>=2.x
numpy
matplotlib
Pillow
opencv-python
tensorboard
```

(El notebook monta Google Drive con `google.colab`, así que localmente habría que cambiar las rutas.)

## Resultados

Evaluación en el conjunto de validación (32×32 → 64×64):

| Método | PSNR (dB) | SSIM |
|---|---|---|
| Bicúbica | **[COMPLETAR]** | **[COMPLETAR]** |
| SRGAN | **[COMPLETAR]** | **[COMPLETAR]** |

<!-- Agrega aquí las curvas de entrenamiento y las comparaciones visuales LR | SR | HR -->

## Limitaciones

- El PSNR y el SSIM premian la similitud píxel a píxel; una interpolación bicúbica puede igualar a la SRGAN en estas métricas aunque visualmente se vea más borrosa.
- El peso adversarial (1e-4) es el estándar de SRGAN, pero hace que el componente adversarial influya poco frente a la pérdida perceptual.
- Las imágenes son pequeñas (64×64), lo que limita el detalle fino disponible para aprender.
- No hay conjunto de prueba independiente: el de validación también se usa para elegir el mejor modelo.
- La demo 64×64 → 128×128 solo se evalúa visualmente, porque no existe imagen de referencia de 128×128.

## Referencias

1. I. Goodfellow *et al.*, "Generative adversarial nets," *NeurIPS*, 2014.
2. C. Ledig *et al.*, "Photo-realistic single image super-resolution using a generative adversarial network," *CVPR*, 2017.
3. K. He *et al.*, "Deep residual learning for image recognition," *CVPR*, 2016.
4. W. Shi *et al.*, "Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network," *CVPR*, 2016.
5. K. Simonyan y A. Zisserman, "Very deep convolutional networks for large-scale image recognition," *ICLR*, 2015.

## Autor

**Juan Camilo Henao Espinosa** — Universidad Autónoma de Occidente
