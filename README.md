# EcoClasifica

Proyecto de entrenamiento en IA orientado a la clasificación de residuos según el material del objeto, con una lógica sencilla pero útil para identificar categorías como plástico, papel, vidrio, metal u otros.

## Descripción

EcoClasifica es un módulo Python que recibe una descripción textual de un residuo y devuelve su categoría probable junto con un nivel de confianza. Actualmente, el proyecto implementa una versión inicial basada en reglas y alias de palabras clave, ideal como base para un sistema más avanzado de reconocimiento visual o de texto.

## Objetivo

Ayudar a automatizar la separación de residuos y fomentar la reciclabilidad mediante una clasificación rápida por tipo de material.

## Categorías soportadas

- Plástico
- Papel
- Vidrio
- Metal
- Otros

## Estructura del proyecto

```text
Proyecto_Entrenamiento_IA/
├── README.md
├── requirements.txt
├── plan-ecoclasifica.txt
├── src/
│   ├── __init__.py
│   └── ecoclasifica.py
├── tests/
│   ├── __init__.py
│   └── test_ecoclasifica.py
└── htmlcov/
    └── ... archivos de cobertura
```

## Requisitos

- Python 3.10+
- pip

## Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/MarcoBonifacio/Proyecto_Entrenamiento_IA.git
cd Proyecto_Entrenamiento_IA
```

2. Crea un entorno virtual:

```bash
python -m venv venv
```

3. Activa el entorno:

En Windows:

```bash
venv\Scripts\activate
```

En macOS/Linux:

```bash
source venv/bin/activate
```

4. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

Puedes importar la función principal desde el paquete `src`:

```python
from src.ecoclasifica import classify_residue

resultado = classify_residue("Botella de vidrio")
print(resultado)
# {'category': 'glass', 'confidence': 0.9}
```

También puedes pasar el tamaño de la imagen para reducir la confianza cuando la resolución es baja:

```python
from src.ecoclasifica import classify_residue

resultado = classify_residue("Envase de plastico", image_size=(64, 64))
print(resultado)
```

## Funciones principales

- `normalize_material_name(description: str) -> str`
  - Normaliza la descripción y devuelve la categoría detectada.

- `calculate_confidence(category: str) -> float`
  - Retorna la confianza base de la categoría.

- `classify_residue(description: str, image_size: tuple[int, int] | None = None) -> dict`
  - Clasifica el residuo y ajusta la confianza si la imagen es pequeña.

## Pruebas

El proyecto incluye pruebas unitarias con `unittest`.

Ejecuta:

```bash
python -m unittest discover -s tests -v
```

## Cobertura

Se puede generar cobertura con:

```bash
coverage run -m unittest discover -s tests -v
coverage report -m
```

## Notas

Este repositorio representa una base de trabajo para un sistema de clasificación de residuos. En su estado actual, la lógica es determinista y basada en reglas textuales, pero puede servir como punto de partida para una versión con visión artificial, modelos de machine learning o integración con una interfaz web.

## Licencia

Proyecto de entrenamiento académico. Revisa el contenido del repositorio antes de usarlo en producción o en entornos comerciales.
