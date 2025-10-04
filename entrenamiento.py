# Importar librerías
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from prettytable import PrettyTable

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

# Realizar predicciones sobre el conjunto de prueba escalado
y_pred = model.predict(X_test)

# Calcula métricas de rendimiento del modelo Random Forest
accuracy = accuracy_score(y_test, y_pred)*100.0
f1 = f1_score(y_test, y_pred)*100.0
precision = precision_score(y_test, y_pred)*100.0
recall = recall_score(y_test, y_pred)*100.0
auc = roc_auc_score(y_test, y_pred)*100.0

# Tabla con métricas
to_show = PrettyTable(['Modelo', 'Exactitud', 'F1-Score', 'Precisión', 'Recall', 'AUC'])
to_show.add_row(['Random Forest', f'{accuracy:.5f}', f'{f1:.5f}', f'{precision:.5f}', f'{recall:.5f}', f'{auc:.5f}'])
print(to_show)