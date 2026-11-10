# EcoClasifica — Clasificación Inteligente de Residuos

Sistema de visión artificial basado en Deep Learning y Transfer Learning para clasificar residuos sólidos a partir de imágenes, facilitando la separación ecológica y el reciclaje automatizado.

---

## Equipo de Desarrollo (Equipo DEV)

- **Bryan Alexander Cahuana Mendoza** — DevOps / CI-CD
- **Marco Antonio Bonifacio Ruiz** — Ingeniero ML / IA
- **Jerson Alejandro Baca Medina** — Desarrollador Fullstack
- **Yury Yohan Llanos Luna** — Desarrollador Backend
- **Benjamin Salazar Arenas** — Desarrollador Frontend

Proyecto Final — Construcción de Software — **Universidad Continental**

---

## Problema y Solución

- **Problema:** La dificultad de reconocer el material de un residuo para separarlo adecuadamente genera contaminación cruzada en puntos limpios y reduce la tasa de aprovechamiento de materiales reciclables.
- **Solución:** Una aplicación web con interfaz gráfica interactiva (Gradio) que analiza fotografías de residuos, clasifica el tipo de material mediante la arquitectura ResNet-50 y ofrece recomendaciones de disposición ecológica.

---

## Arquitectura de IA y Métricas

- **Categoría:** Classification
- **Modelo Base:** microsoft/resnet-50 (preentrenado en ImageNet-1k)
- **Técnica:** Transfer Learning y Fine-Tuning
- **Resolución de Entrada:** Imágenes RGB redimensionadas a 224 x 224 píxeles
- **Dispositivo de Entrenamiento:** CPU local
- **Resultado del Entrenamiento (5 Épocas):**
  - **Accuracy (Evaluación):** ~58.42%
  - **F1 Macro (Evaluación):** ~0.4709
  - **Ubicación de Pesos:** models/ecoclasifica-resnet50/

### Categorías Soportadas (6 Clases):
1. Cartón (`cardboard`)
2. Vidrio (`glass`)
3. Metal (`metal`)
4. Papel (`paper`)
5. Plástico (`plastic`)
6. Basura general (`trash`)

---

## Dataset

- **Origen:** TrashNet (dataset-resized)
- **Volumen Total:** 2,527 imágenes reales distribuidas en 6 categorías
- **Partición:** 80% Entrenamiento (~2,021 imágenes) / 20% Validación (~506 imágenes)

---

## Estructura del Repositorio

```text
Proyecto_Entrenamiento_IA/
├── README.md                  # Documentación integral del proyecto
├── requirements.txt           # Dependencias del proyecto
├── .gitignore                 # Exclusión de archivos pesados (models/, data/, venv/)
├── preparar_datos.py          # Script de búsqueda, extracción y partición 80/20 de datos
├── src/
│   └── app.py                 # Aplicación principal e interfaz web Gradio
├── training/
│   ├── config.json            # Configuración de hiperparámetros
│   └── train.py               # Pipeline de entrenamiento con Hugging Face Trainer
├── data/
│   └── test_dataset/          # Dataset estructurado en train y validation (ignorado en Git)
└── models/
    └── ecoclasifica-resnet50/ # Pesos entrenados y mapeo de etiquetas (ignorado en Git)
Guía de Instalación y Ejecución Paso a Paso
1. Clonar el Repositorio
Bash
git clone [https://github.com/MarcoBonifacio/Proyecto_Entrenamiento_IA.git](https://github.com/MarcoBonifacio/Proyecto_Entrenamiento_IA.git)
cd Proyecto_Entrenamiento_IA
2. Crear y Activar Entorno Virtual
En Windows PowerShell:

PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
En Linux / macOS:

Bash
python3 -m venv venv
source venv/bin/activate
3. Instalar Dependencias
PowerShell
pip install --upgrade pip
pip install -r requirements.txt
4. Extraer y Estructurar el Dataset
Descarga el archivo ZIP del dataset TrashNet (dataset-resized.zip), extráelo en tu carpeta de Descargas y ejecuta el script automatizado para crear la estructura 80/20 en data/test_dataset:

PowerShell
python preparar_datos.py
5. Entrenar el Modelo (Opcional)
Si deseas repetir el entrenamiento de 5 épocas y generar nuevamente los pesos en models/ecoclasifica-resnet50:

PowerShell
python training/train.py --epochs 5
6. Iniciar la Interfaz Web (Gradio)
Para realizar inferencias en tiempo real utilizando los pesos preentrenados locales:

PowerShell
python src/app.py
Abre en tu navegador la dirección local: http://127.0.0.1:7860

Control de Versiones
Las carpetas de gran tamaño (data/, models/, venv/) se encuentran adecuadamente parametrizadas dentro del archivo .gitignore para evitar la subida de binarios pesados al repositorio remoto.

---

## 📥 Guía de Descarga del Modelo Entrenado

Si ya clonaste el repositorio e instalaste las dependencias en tu entorno virtual, sigue estos pasos para incorporar los pesos entrenados del modelo y ejecutar la aplicación sin tener que entrenar nuevamente:

### 1. Descargar la carpeta de pesos desde Google Drive
Accede al siguiente enlace público de Google Drive y descarga la carpeta completa de pesos:
- **Enlace de descarga:** [Descargar modelo ecoclasifica-resnet50](https://drive.google.com/drive/folders/1eoQL2Vtrk1EnGlQPMl_FayCfSkH47oEZ?usp=drive_link)

### 2. Ubicar la carpeta en el proyecto
Mueve la carpeta descargada `ecoclasifica-resnet50` dentro de la ruta `models/` de tu proyecto local. La estructura debe quedar exactamente así:

```text
Proyecto_Entrenamiento_IA/
└── models/
    └── ecoclasifica-resnet50/
        ├── config.json
        ├── label_mapping.json
        ├── model.safetensors
        └── preprocessor_config.json

3. Iniciar la Aplicación
Una vez colocada la carpeta en la ruta indicada, ejecuta el servidor web de Gradio desde tu terminal en PowerShell:

PowerShell
python src/app.py
Abre tu navegador e ingresa a http://127.0.0.1:7860 para comenzar a clasificar imágenes en tiempo real.
