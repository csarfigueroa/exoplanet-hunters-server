# Sistema de Persistencia del Modelo

## Descripción

El modelo de clasificación de exoplanetas se guarda automáticamente en disco después de cada entrenamiento y se carga automáticamente al iniciar el servidor.

## Funcionamiento

### 1. Guardado Automático

Cuando entrenas el modelo usando `POST /exoplanet/train`, el sistema:

1. Entrena el modelo con los datos proporcionados
2. **Guarda automáticamente** el modelo en `saved_models/exoplanet_model.pth`
3. **Guarda automáticamente** el scaler en `saved_models/scaler.pkl`
4. Retorna en la respuesta: `"model_saved": true`

**No necesitas hacer nada adicional** - el guardado es automático.

### 2. Carga Automática

Cuando el servidor se inicia (o reinicia):

1. Busca archivos del modelo en `saved_models/`
2. Si existen, **carga automáticamente** el modelo y el scaler
3. El modelo queda listo para hacer predicciones inmediatamente
4. Verás en los logs:

```
🔄 Cargando modelo existente desde saved_models/exoplanet_model.pth...
✓ Modelo cargado exitosamente
  - Training accuracy: 92.45%
  - Test accuracy: 85.67%
  - Features: 10
```

### 3. Sin Modelo Guardado

Si no hay modelo guardado, verás:

```
ℹ No se encontró modelo guardado. Entrena el modelo con POST /exoplanet/train
```

## Archivos Guardados

Los siguientes archivos se crean en `saved_models/`:

| Archivo | Contenido | Tamaño Aprox. |
|---------|-----------|---------------|
| `exoplanet_model.pth` | Pesos de la red neuronal, métricas, features | ~500 KB |
| `scaler.pkl` | Normalizador de features (StandardScaler) | ~5 KB |

## Persistencia en Docker

### Configuración Actual

El `docker-compose.yml` ya tiene configurado el volumen:

```yaml
volumes:
  - .:/app
```

Esto significa que:
- ✅ Los modelos persisten entre reinicios del contenedor
- ✅ Los modelos persisten cuando actualizas el código
- ✅ Los modelos están disponibles en tu máquina local

### Ubicación de los Archivos

**Dentro del contenedor:**
```
/app/saved_models/
```

**En tu máquina local:**
```
/Users/cesarfigueroa/Documents/exoplanet-hunters-server/saved_models/
```

## Flujo de Trabajo

### Primera Vez (Sin Modelo)

1. Inicia el servidor: `docker-compose up`
2. Entrena el modelo: `POST /exoplanet/train`
3. El modelo se guarda automáticamente
4. Haz predicciones: `POST /exoplanet/predict`

### Reinicios Posteriores

1. Reinicia el servidor: `docker-compose restart`
2. El modelo se carga automáticamente
3. Haz predicciones inmediatamente (sin necesidad de re-entrenar)

### Actualizar el Modelo

Si quieres re-entrenar con nuevos datos o parámetros:

1. `POST /exoplanet/train` con nuevos parámetros
2. El modelo anterior se **sobrescribe** automáticamente
3. El nuevo modelo queda disponible inmediatamente

## Verificar Estado del Modelo

### Endpoint de Status

```bash
GET /exoplanet/status
```

**Respuesta si el modelo está cargado:**
```json
{
  "model_trained": true,
  "training_accuracy_percentage": 92.45,
  "test_accuracy_percentage": 85.67,
  "input_features": 10,
  "objective": "Binary classification: Confirmed Exoplanet vs Non-Confirmed",
  "data_source": "TESS Objects of Interest (TOI) catalog"
}
```

**Respuesta si NO hay modelo:**
```json
{
  "model_trained": false,
  "model_exists": false,
  "message": "Model not trained yet. Use /train endpoint to train the model."
}
```

## Información Guardada

El archivo `exoplanet_model.pth` contiene:

```python
{
  'model_state_dict': ...     # Pesos de la red neuronal
  'input_size': 10,           # Número de features
  'training_accuracy': 0.9245, # Accuracy en entrenamiento
  'test_accuracy': 0.8567,    # Accuracy en test
  'feature_columns': [...]    # Lista de features usados
}
```

## Gestión Manual

### Ver Archivos del Modelo

```bash
# Dentro del contenedor
docker-compose exec neural-network-api ls -lh saved_models/

# En tu máquina local
ls -lh saved_models/
```

### Eliminar Modelo Guardado

```bash
# Para forzar re-entrenamiento
rm -rf saved_models/

# O dentro del contenedor
docker-compose exec neural-network-api rm -rf saved_models/
```

### Backup del Modelo

```bash
# Crear backup
cp -r saved_models/ saved_models_backup_$(date +%Y%m%d)/

# Restaurar backup
cp -r saved_models_backup_20241004/ saved_models/
```

## Versionado

Los modelos **NO** están en Git (ver `.gitignore`). Esto es intencional porque:

- Los modelos son grandes (~500 KB)
- Se pueden regenerar entrenando
- Cambian frecuentemente durante desarrollo

Si necesitas versionar modelos:

```bash
# Crear versión específica
cp saved_models/exoplanet_model.pth models_archive/model_v1.0.pth
git add models_archive/model_v1.0.pth
git commit -m "Release model v1.0"
```

## Troubleshooting

### Problema: "Model not trained yet" después de entrenar

**Causa:** El modelo no se guardó correctamente

**Solución:**
1. Verifica que existe `saved_models/` en el contenedor
2. Revisa los logs durante el entrenamiento
3. Asegúrate de que el volumen de Docker está montado

### Problema: Modelo se resetea al reiniciar

**Causa:** El volumen de Docker no está persistiendo

**Solución:**
```bash
# Verifica el volumen en docker-compose.yml
volumes:
  - .:/app  # Debe estar presente

# Reconstruye el contenedor
docker-compose down
docker-compose up --build
```

### Problema: Error al cargar modelo antiguo

**Causa:** Cambios en la arquitectura del modelo

**Solución:**
```bash
# Elimina el modelo antiguo y re-entrena
rm -rf saved_models/
POST /exoplanet/train
```

## Logs Útiles

Durante el inicio del servidor:
```
✓ Modelo cargado exitosamente
  - Training accuracy: 92.45%
  - Test accuracy: 85.67%
```

Durante el entrenamiento:
```
Guardando modelo entrenado...
✓ Modelo guardado en saved_models/exoplanet_model.pth
✓ Scaler guardado en saved_models/scaler.pkl
```

## Mejores Prácticas

1. ✅ **Entrena una vez**, úsalo muchas veces
2. ✅ **Verifica el status** antes de hacer predicciones
3. ✅ **Haz backup** de modelos que funcionan bien
4. ✅ **Re-entrena** solo cuando agregues nuevos datos
5. ✅ **Documenta** los parámetros de entrenamiento usados

## Ejemplo Completo

```bash
# 1. Iniciar servidor
docker-compose up -d

# 2. Verificar estado (no hay modelo)
curl http://localhost:8000/exoplanet/status
# => "model_trained": false

# 3. Entrenar modelo
curl -X POST http://localhost:8000/exoplanet/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 100, "learning_rate": 0.001}'
# => "model_saved": true

# 4. Verificar que se guardó
ls saved_models/
# => exoplanet_model.pth  scaler.pkl

# 5. Reiniciar servidor
docker-compose restart

# 6. Verificar que se cargó automáticamente
curl http://localhost:8000/exoplanet/status
# => "model_trained": true

# 7. Hacer predicción (sin re-entrenar)
curl -X POST http://localhost:8000/exoplanet/predict \
  -H "Content-Type: application/json" \
  -d @test_samples/sample_1.json
# => Predicción exitosa
```
