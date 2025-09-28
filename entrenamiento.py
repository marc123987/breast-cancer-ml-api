# Importar librerías
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Nombre del archivo
file_path = "data.csv"

# Se carga el dataset
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "uciml/breast-cancer-wisconsin-data",
  file_path,
)

# Se elimina la columna con NaN
df = df.drop('Unnamed: 32', axis=1)
# Se elimina columna que no aporta información
df = df.drop('id', axis=1)

# Separación de variable objetivo
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Codificación de variable categórica
label_encoder = LabelEncoder()
categorical_y = label_encoder.fit_transform(y)
print("Objetivo codificado\n---------------")
display(categorical_y)

# Divide conjunto de entrenamiento (80%) y prueba(20%)
X_train, X_test, y_train, y_test = train_test_split(X, categorical_y, test_size=0.2, random_state=42, stratify=categorical_y)

# Entrenamiento del modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Guardar modelo
joblib.dump(model, 'modelo_cancer.pkl')

print("Modelo entrenado y guardado como 'modelo_cancer.pkl'")