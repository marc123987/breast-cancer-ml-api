# Breast Cancer ML Model API

Este proyecto implementa un modelo de Machine Learning para la clasificación del dataset Breast Cancer usando Random Forest, con una API REST construida con Flask y configuración para despliegue en Kubernetes.

## 📋 Descripción

El proyecto incluye:
- **Modelo de ML**: Random Forest Classifier entrenado con el dataset de Breast Cancer
- **API REST**: Endpoint Flask para realizar predicciones
- **Containerización**: Dockerfile para crear imagen Docker
- **Kubernetes**: Archivos YAML para despliegue en K8s

## 📋 Objetivo

Desarrollar un sistema completo que integre un modelo de Machine Learning, exponerlo como API REST mediante Flask y contenedorizado con Docker, automatizando el flujo de trabajo usando buenas prácticas de CI/CD.

## 🚀 Estructura del Proyecto

```
├── entrenamiento.py      # Script para entrenar el modelo
├── modelo_cancer.pkl     # Modelo entrenado serializado
├── api.py             # API Flask con endpoint /predict
├── test_api.py        # Script de pruebas para la API
├── requirements.txt   # Dependencias del proyecto
├── Dockerfile         # Imagen Docker
├── deployment.yaml    # Deployment de Kubernetes
├── service.yaml       # Servicio de Kubernetes
└── README.md          # Este archivo
```

## 🛠️ Instalación y Uso Local

### Prerrequisitos
- Python 3.7+
- pip

### 1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd <nombre-del-repositorio>
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Entrenar el modelo (opcional)
El modelo ya está entrenado, pero puedes re-entrenarlo:
```bash
python modelo_cancer.py
```

### 4. Ejecutar la API
```bash
python api.py
```

La API estará disponible en `http://localhost:5000`

### 5. Probar la API (opcional)
```bash
python test_api.py
```

## 📡 API Usage

### Endpoint: POST /predict

Realiza predicciones sobre nuevos datos de características de los núcleos celulares para predecir si una paciente tiene cáncer mamario.

**URL**: `http://localhost:5000/predict`

**Método**: `POST`

**Parámetros del cuerpo (JSON)**:
```json
{
    "features": [17.990000,10.380000,122.800000,1001.000000,0.118400,0.277600,0.300100,0.147100,0.241900,0.078710,1.095000,0.905300,8.589000,153.400000,0.006399,0.049040,0.053730,0.015870,0.030030,0.006193,25.380000,17.330000,184.600000,2019.000000,0.162200,0.665600,0.711900,0.265400,0.460100,0.118900]
}
```

Los features corresponden a:
- `radius_mean`
- `texture_mean`
- `perimeter_mean`
- `area_mean`
- `smoothness_mean`
- `compactness_mean`
- `concavity_mean`
- `concave_points_mean`
- `symmetry_mean`
- `fractal_dimension_mean`
- `radius_se`
- `texture_se`
- `perimeter_se`
- `area_se`
- `smoothness_se`
- `compactness_se`
- `concavity_se`
- `concave_points_se`
- `symmetry_se`
- `fractal_dimension_se`
- `radius_worst`
- `texture_worst`
- `perimeter_worst`
- `area_worst`
- `smoothness_worst`
- `compactness_worst`
- `concavity_worst`
- `concave_points_worst`
- `symmetry_worst`
- `fractal_dimension_worst`

**Respuesta exitosa**:
```json
{
    "prediction": 1
}
```

**Respuesta de error**:
```json
{
    "error": "Mensaje de error"
}
```

### Clases de predicción:
- `0`: Maligno
- `1`: Benigno

### Ejemplo con curl:
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [17.990000,10.380000,122.800000,1001.000000,0.118400,0.277600,0.300100,0.147100,0.241900,0.078710,1.095000,0.905300,8.589000,153.400000,0.006399,0.049040,0.053730,0.015870,0.030030,0.006193,25.380000,17.330000,184.600000,2019.000000,0.162200,0.665600,0.711900,0.265400,0.460100,0.118900]}'
```

## 🐳 Docker

### Construir imagen
```bash
docker build -t breast-cancer-ml-api .
```

### Ejecutar contenedor
```bash
docker run -p 5000:5000 breast-cancer-ml-api
```

## ☸️ Despliegue en Kubernetes

## 🔄 CI/CD: Flujo Simplificado de Deploy

Se ha configurado un pipeline de GitHub Actions en `.github/workflows/ci-cd.yml` que demuestra un flujo completo de CI/CD para modelos de ML:

### 🔄 Proceso automatizado

**En cada push a `main`:**

1. **Tests** 🧪
   - Instala dependencias Python
   - Lanza la API en background
   - Prueba endpoint `/`
   - Prueba endpoint `/predict` con datos reales
   - Termina la API limpiamente

2. **Build & Deploy** 🚀 (solo si tests pasan)
   - Construye imagen Docker
   - Sube a Docker Hub con dos tags:
     - `latest`
     - `<commit-sha>` (trazabilidad)

### ⚙️ Configuración requerida

En tu repositorio de GitHub: **Settings > Secrets > Actions**

Crea estos secretos:
- `DOCKERHUB_USERNAME`: tu usuario de Docker Hub  
- `DOCKERHUB_TOKEN`: token de acceso (Docker Hub > Account Settings > Security)

### ▶️ Uso

```bash
# Ejecución manual desde GitHub Actions tab
# O simplemente: git push origin main

# Descargar imagen publicada
docker pull tu_usuario/breast-cancer-ml-api:latest
docker run -p 5000:5000 tu_usuario/breast-cancer-ml-api:latest
```

### 📚 Valor educativo

Este pipeline enseña:
- Tests de integración básicos con curl
- Manejo de procesos en background en CI
- Docker build y publish automático  
- Dependencias entre jobs (`needs: tests`)
- Gestión de secretos en GitHub Actions


### 1. Aplicar deployment
```bash
kubectl apply -f deployment.yaml
```

### 2. Aplicar servicio
```bash
kubectl apply -f service.yaml
```

### 3. Verificar despliegue
```bash
kubectl get pods
kubectl get services
```

## 🧪 Pruebas de Ejemplo

Puedes probar diferentes predicciones (para tumor benigno o maligno):

```bash
# Tumor benigno (esperado: 0)
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [7.760000,24.540000,47.920000,181.000000,0.052630,0.043620,0.000000,0.000000,0.158700,0.058840,0.385700,1.428000,2.548000,19.150000,0.007189,0.004660,0.000000,0.000000,0.026760,0.002783,9.456000,30.370000,59.160000,268.600000,0.089960,0.064440,0.000000,0.000000,0.287100,0.070390]}'

# Tumor maligno (esperado: 1) 
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [17.990000,10.380000,122.800000,1001.000000,0.118400,0.277600,0.300100,0.147100,0.241900,0.078710,1.095000,0.905300,8.589000,153.400000,0.006399,0.049040,0.053730,0.015870,0.030030,0.006193,25.380000,17.330000,184.600000,2019.000000,0.162200,0.665600,0.711900,0.265400,0.460100,0.118900]}'
```

## 📚 Tecnologías Utilizadas

- **Python 3.x**
- **scikit-learn**: Machine Learning
- **Flask**: API REST
- **joblib**: Serialización del modelo
- **numpy**: Computación numérica
- **Docker**: Containerización
- **Kubernetes**: Orquestación de contenedores

## 📝 Notas

- El modelo Random Forest fue entrenado con 80% del dataset completo de Breast cancer (455 muestras)
- Este proyecto es con fines educativos y demostrativos

## 📝 Reflexiones de lo aprendido

En este proyecto se implementó un proceso de CI/CD mediante la plataforma de GitHub Actions y Docker Hub para aprovechar las ventajas de utilizar una imagen de Docker con un ambiente que sea consistente para la correcta compilación y ejecución de la aplicación, independiente de la infraestructura de los usuarios que hagan uso del contenedor generado a partir de dicha imagen.

Además de contar con el control de versiones que provee GitHub para los archivos de código, los archivos con configuraciones de pipelines y otros archivos, así como GitHub Actions que se encarga de compilar, probar y desplegar el código y archivos de la aplicación en la imagen de Docker Hub de forma automatizada, así que es más eficiente ya que así se puede verificar que los cambios subidos no afecten negativamente el comportamiento de la aplicación gracias a las pruebas automatizadas y el despliegue se realiza de forma más rápida.gracias a los jobs o etapas definidas en el pipeline.