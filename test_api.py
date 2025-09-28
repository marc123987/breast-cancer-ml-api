#!/usr/bin/env python3
"""Script de prueba para verificar la API de Breast Cancer ML."""
import requests

# Configuración
BASE_URL = "http://localhost:5000"

def test_get():
    """Probar el endpoint del estado de servicio"""
    print("🔍 Probando estado de servicio...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se puede conectar a la API. ¿Está ejecutándose?")
        return False

def test_predict():
    """Probar el endpoint de predicción"""
    print("\n🔍 Probando predicción...")
    
    # Datos de prueba
    test_data = {
        "features": [17.990000,10.380000,122.800000,1001.000000,0.118400,0.277600,0.300100,0.147100,0.241900,0.078710,1.095000,0.905300,8.589000,153.400000,0.006399,0.049040,0.053730,0.015870,0.030030,0.006193,25.380000,17.330000,184.600000,2019.000000,0.162200,0.665600,0.711900,0.265400,0.460100,0.118900]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se puede conectar a la API.")
        return False


def test_invalid_data():
    """Probar con datos inválidos"""
    print("\n🔍 Probando validación de datos...")
    
    # Datos inválidos (solo 2 features)
    invalid_data = {
        "features": [10, 2.5]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 400
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se puede conectar a la API.")
        return False


if __name__ == "__main__":
    print("🚀 Iniciando pruebas de la API...")
    print("=" * 50)
    
    # Ejecutar pruebas
    get_ok = test_get()
    predict_ok = test_predict()
    validation_ok = test_invalid_data()
    
    # Resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS:")
    print(f"✅ Get Check: {'PASS' if get_ok else 'FAIL'}")
    print(f"✅ Predicción: {'PASS' if predict_ok else 'FAIL'}")
    print(f"✅ Validación: {'PASS' if validation_ok else 'FAIL'}")
    
    if all([get_ok, predict_ok, validation_ok]):
        print("\n🎉 ¡Todas las pruebas pasaron!")
    else:
        print("\n⚠️  Algunas pruebas fallaron.")

