# 🌿 EcoClasifica — Clasificación Inteligente de Residuos

Sistema inteligente de visión artificial basado en Deep Learning y Transfer Learning para clasificar residuos sólidos a partir de imágenes, facilitando la separación ecológica y el reciclaje automatizado.

---

## 👥 Equipo de Desarrollo (Equipo DEV)

* **Bryan Alexander Cahuana Mendoza** — *DevOps / CI-CD*
* **Marco Antonio Bonifacio Ruiz** — *Ingeniero ML / IA*
* **Jerson Alejandro Baca Medina** — *Desarrollador Fullstack*
* **Yury Yohan Llanos Luna** — *Desarrollador Backend*
* **Benjamin Salazar Arenas** — *Desarrollador Frontend*

*Proyecto Final — Construcción de Software — **Universidad Continental***

---

## 📌 Problema y Solución

* **Problema:** La dificultad de reconocer el material de un residuo para separarlo adecuadamente genera contaminación cruzada en puntos limpios y reduce la tasa de aprovechamiento de materiales reciclables.
* **Solución:** Una aplicación multiplataforma con interfaz gráfica interactiva (**Gradio**) que analiza fotografías de residuos individuales, clasifica el tipo de material con probabilidades estimadas y brinda recomendaciones inmediatas de disposición ecológica.

---

## 🧠 Arquitectura de IA

* **Categoría:** `image-classification`
* **Modelo Base:** [`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50) (pre-entrenado en ImageNet-1k)
* **Técnica:** Transfer Learning y Fine-Tuning adaptado a 6 clases de residuos
* **Resolución de Entrada:** Imágenes RGB redimensionadas a 224 × 224 píxeles
* **Salida:** Categoría predicha con mayor probabilidad y distribución softmax completa de clases
* **Justificación:** ResNet-50 ofrece un balance óptimo entre precisión top-1, extracción de características visuales profundas mediante conexiones residuales y eficiencia computacional para inferencia local en tiempo real.

### Categorías Soportadas (6 clases):
1. 📦 **Cartón** (`cardboard`)
2. 🫙 **Vidrio** (`glass`)
3. 🥫 **Metal** (`metal`)
4. 📄 **Papel** (`paper`)
5. ♻️ **Plástico** (`plastic`)
6. 🗑️ **Basura general** (`trash`)

---

## 📊 Dataset

* **Nombre:** [`TrashBusters/combinedd`](https://huggingface.co/datasets/TrashBusters/combinedd)
* **Cantidad de Registros:** 29,849 imágenes etiquetadas
* **Columnas:** `image`, `label`, `label_id`, `label_name`, `source_dataset`, `original_label`, `image_context`
* **División:** 80% Entrenamiento, 10% Validación, 10% Prueba

---

## 📁 Estructura del Repositorio

```text
ecoclasifica/
├── README.md                  # Documentación integral del proyecto
├── requirements.txt           # Dependencias del stack tecnológico
├── .gitignore                 # Exclusiones de Git (modelos pesados, caché)
├── .github/
│   └── workflows/
│       └── ci.yml             # Automatización CI/CD con GitHub Actions
├── src/                       # Código fuente de la aplicación
│   ├── __init__.py            # Inicializador de paquete
│   ├── preprocess.py          # Validación y preprocesamiento de imágenes (224x224)
│   ├── model.py               # Clase EcoClasificaModel e inferencia ResNet-50
│   ├── postprocess.py         # Formateo de probabilidades y consejos ecológicos
│   ├── app.py                 # Interfaz interactiva Gradio
│   ├── routes.py              # Controladores y servicios modulares
│   └── api.py                 # Envoltorio de endpoints para integración
├── tests/                     # Suite de pruebas unitarias (TDD)
│   ├── __init__.py
│   ├── test_preprocess.py     # Pruebas de carga, formatos y normalización
│   ├── test_model.py          # Pruebas del modelo (arquitectura y mocks)
│   ├── test_postprocess.py    # Pruebas de formato y recomendaciones
│   ├── test_api.py            # Pruebas de rutas y endpoints
│   └── test_ecoclasifica.py   # Pruebas de lógica base de reglas
├── training/                  # Pipeline de entrenamiento y evaluación
│   ├── __init__.py
│   ├── config.json            # Hiperparámetros (épocas, LR, batch size)
│   ├── train.py               # Script de entrenamiento con Hugging Face Trainer
│   └── evaluate.py            # Métricas, matriz de confusión y reporte de clasificación
├── data/
│   ├── raw/                   # Datos crudos (.gitkeep)
│   └── processed/             # Datos preparados (.gitkeep)
├── models/                    # Pesos afinados (ignorado en git)
├── static/                    # Recursos visuales y gráficos generados
├── templates/                 # Plantillas HTML adicionales
└── docs/                      # Documentación y reportes adicionales
```

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.10+
* **Framework Web:** Gradio
* **Machine Learning / Visión:** Hugging Face Transformers, PyTorch, Torchvision
* **Dataset Management:** Hugging Face Datasets
* **Métricas y Análisis:** Scikit-Learn, Matplotlib, Seaborn
* **Testing:** `unittest` + TDD
* **Integración Continua:** GitHub Actions CI/CD

---

## 🚀 Instalación y Ejecución Local

### 1. Clonar el Repositorio
```bash
git clone https://github.com/MarcoBonifacio/Proyecto_Entrenamiento_IA.git
cd Proyecto_Entrenamiento_IA
```

### 2. Crear y Activar Entorno Virtual
En Windows:
```bash
py -m venv venv
venv\Scripts\activate
```

En Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

---

## 🧪 Ejecución de Pruebas Unitarias (TDD)

Para ejecutar la batería completa de tests unitarios:
```bash
python -m unittest discover -s tests -v
```

Con reporte de cobertura de código:
```bash
coverage run -m unittest discover -s tests -v
coverage report -m
```

---

## 🖥️ Ejecución de la Interfaz Web (Gradio)

Para iniciar la aplicación gráfica localmente:
```bash
python src/app.py
```
Abre en tu navegador la URL: `http://localhost:7860`

*Características de la interfaz:*
* Subida de imágenes por arrastrar/soltar o captura en directo desde la webcam.
* Gráfico de barras de probabilidad en tiempo real para las 6 categorías.
* Tarjeta informativa con recomendaciones sobre el color de contenedor correspondiente y consejos ecológicos.

---

## 🏋️ Pipeline de Entrenamiento

Para entrenar el modelo ResNet-50 sobre el dataset `TrashBusters/combinedd`:
```bash
python training/train.py
```
*Opcionalmente se pueden sobrescribir hiperparámetros:*
```bash
python training/train.py --epochs 3 --batch_size 16 --lr 5e-5
```

Para evaluar el modelo afinado y generar la matriz de confusión:
```bash
python training/evaluate.py
```

---

## 🔄 Integración Continua (CI/CD)

El repositorio incluye un workflow de GitHub Actions en `.github/workflows/ci.yml` que en cada `push` o `pull request` en ramas `main` y `develop`:
1. Configura matrices de entorno en Python 3.10 y 3.11.
2. Instala dependencias y herramientas de calidad.
3. Ejecuta la suite de pruebas automatizadas con `unittest`.
4. Realiza análisis estático de código mediante `flake8`.
