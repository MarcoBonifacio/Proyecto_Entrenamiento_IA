# EcoClasifica — Clasificación Inteligente de Residuos

Sistema de visión artificial basado en Deep Learning y Transfer Learning para clasificar residuos sólidos a partir de imágenes, facilitando la separación ecológica y el reciclaje automatizado mediante una arquitectura moderna desacoplada (**FastAPI + React**).

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
- **Solución:** Una aplicación web moderna con arquitectura cliente-servidor que analiza fotografías de residuos en tiempo real mediante **ResNet-50** servido a través de una API REST en **FastAPI**, ofreciendo una interfaz interactiva en **React** con diseño *Glassmorphism*, escáner óptico animado, barra de inferencia progresiva, recomendación palabra por palabra (*Typewriter*) y soporte para arrastrar o pegar imágenes directamente desde el portapapeles (`Ctrl + V`).

---

## Stack Tecnológico

- **Inteligencia Artificial / ML:** PyTorch, Torchvision, Hugging Face Transformers (`microsoft/resnet-50`), Pillow, Scikit-learn.
- **Backend (API REST):** FastAPI, Uvicorn, Python-Multipart.
- **Frontend (SPA):** React 19, TypeScript, Vite, Tailwind CSS v4, TanStack Query, Axios, Lucide Icons.

---

## Arquitectura de IA y Métricas

- **Categoría:** Classification
- **Modelo Base:** `microsoft/resnet-50` (preentrenado en ImageNet-1k)
- **Técnica:** Transfer Learning y Fine-Tuning
- **Resolución de Entrada:** Imágenes RGB redimensionadas a `224 x 224` píxeles
- **Dispositivo de Entrenamiento:** CPU local
- **Resultado del Entrenamiento (5 Épocas):**
  - **Accuracy (Evaluación):** ~58.42%
  - **F1 Macro (Evaluación):** ~0.4709
  - **Ubicación de Pesos:** `models/ecoclasifica-resnet50/`

### Categorías Soportadas (6 Clases):
1. Cartón (`cardboard`)
2. Vidrio (`glass`)
3. Metal (`metal`)
4. Papel (`paper`)
5. Plástico (`plastic`)
6. Basura general (`trash`)

---

## Dataset

- **Origen:** TrashNet (`dataset-resized`)
- **Volumen Total:** 2,527 imágenes reales distribuidas en 6 categorías
- **Partición:** 80% Entrenamiento (~2,021 imágenes) / 20% Validación (~506 imágenes)

---

## Estructura del Repositorio

```text
Proyecto_Entrenamiento_IA/
├── README.md                      # Documentación integral del proyecto
├── requirements.txt               # Dependencias Python del backend y entrenamiento
├── .gitignore                     # Exclusión de archivos pesados (models/, data/, venv/, node_modules/)
├── preparar_datos.py              # Script de búsqueda, extracción y partición 80/20 de datos
├── src/
│   └── app.py                     # Servidor API REST con FastAPI e inferencia ResNet-50
├── frontend/                      # Aplicación web React + Vite + TypeScript + Tailwind CSS
│   ├── package.json               # Dependencias del frontend
│   ├── public/
│   │   └── bg-eco.jpg             # Fondo botánico de la interfaz Glassmorphism
│   └── src/
│       ├── App.tsx                # Interfaz principal, drag & drop, cámara y pegado Ctrl + V
│       ├── index.css              # Estilos globales y animación del haz de escaneo óptico
│       └── features/
│           └── classification/
│               ├── api/           # Cliente HTTP hacia FastAPI
│               ├── components/    # ResultCard y animación TypewriterText
│               ├── hooks/         # useClassifyImage (estados de pipeline y barra progresiva)
│               └── types/         # Tipado estricto TypeScript
├── training/
│   ├── config.json                # Configuración de hiperparámetros
│   └── train.py                   # Pipeline de entrenamiento con Hugging Face Trainer
├── data/
│   └── test_dataset/              # Dataset estructurado en train y validation (ignorado en Git)
└── models/
    └── ecoclasifica-resnet50/     # Pesos entrenados y mapeo de etiquetas (ignorado en Git)
📥 Guía de Descarga del Modelo Entrenado
Si clonaste el repositorio y deseas ejecutar la aplicación directamente sin volver a entrenar el modelo:

1. Descargar la carpeta de pesos desde Google Drive
Accede al siguiente enlace público y descarga la carpeta completa de pesos:

Enlace de descarga: Descargar modelo ecoclasifica-resnet50

2. Ubicar la carpeta en el proyecto
Mueve la carpeta descargada ecoclasifica-resnet50 dentro de la ruta models/ de tu proyecto local. La estructura debe quedar así:

Plaintext
Proyecto_Entrenamiento_IA/
└── models/
    └── ecoclasifica-resnet50/
        ├── config.json
        ├── label_mapping.json
        ├── model.safetensors
        └── preprocessor_config.json
Guía de Instalación y Ejecución Paso a Paso
1. Clonar el Repositorio
Bash
git clone [https://github.com/MarcoBonifacio/Proyecto_Entrenamiento_IA.git](https://github.com/MarcoBonifacio/Proyecto_Entrenamiento_IA.git)
cd Proyecto_Entrenamiento_IA
2. Configurar y Levantar el Backend (FastAPI)
Crear y activar entorno virtual:

En Windows PowerShell:

PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
En Linux / macOS:

Bash
python3 -m venv venv
source venv/bin/activate
Instalar dependencias de Python:

PowerShell
pip install --upgrade pip
pip install -r requirements.txt
Iniciar el servidor API (Puerto 8000):

PowerShell
python src/app.py
La API quedará escuchando en http://localhost:8000 (documentación interactiva Swagger en http://localhost:8000/docs).

3. Configurar y Levantar el Frontend (React + Vite)
Abre una segunda terminal en la raíz del proyecto, entra a la carpeta frontend/ e instala las dependencias:

PowerShell
cd frontend
npm install
npm run dev
Abre en tu navegador la dirección local: http://localhost:5173 para comenzar a clasificar imágenes en tiempo real (subiendo archivos, usando la cámara, arrastrando imágenes o pegando con Ctrl + V).

Entrenamiento desde Cero (Opcional)
Extraer y Estructurar el Dataset: Descarga dataset-resized.zip (TrashNet) en tu carpeta de Descargas y ejecuta:

PowerShell
python preparar_datos.py
Entrenar el Modelo (5 Épocas):

PowerShell
python training/train.py --epochs 5
Control de Versiones
Las carpetas de gran tamaño y dependencias locales (data/, models/, venv/, frontend/node_modules/, frontend/dist/, .vscode/) se encuentran parametrizadas dentro del archivo .gitignore para mantener el repositorio ligero y limpio.
