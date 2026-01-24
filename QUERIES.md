# Script de Requêtes DuckDB - Prédictions d'Accidents

## Connexion à la base

```bash
duckdb predictions.duckdb
```

## Requêtes Utiles

### Voir les prédictions du lendemain

```sql
SELECT 
    date,
    hour,
    COUNT(*) as nb_points,
    AVG(probability) as risque_moyen,
    MAX(probability) as risque_max,
    COUNT(CASE WHEN probability >= 0.8 THEN 1 END) as points_critique
FROM predictions
WHERE date = CURRENT_DATE + INTERVAL 1 DAY
GROUP BY date, hour
ORDER BY hour;
```

### Top 10 des zones les plus dangereuses pour demain

```sql
SELECT 
    date,
    hour,
    latitude,
    longitude,
    probability,
    ROUND(latitude, 3) || ', ' || ROUND(longitude, 3) as coordonnees
FROM predictions
WHERE date = CURRENT_DATE + INTERVAL 1 DAY
ORDER BY probability DESC
LIMIT 10;
```

### Heure la plus dangereuse par jour

```sql
SELECT 
    date,
    hour,
    AVG(probability) as risque_moyen,
    COUNT(*) as nb_points
FROM predictions
GROUP BY date, hour
HAVING AVG(probability) = (
    SELECT MAX(avg_prob) 
    FROM (
        SELECT AVG(probability) as avg_prob 
        FROM predictions p2 
        WHERE p2.date = predictions.date 
        GROUP BY hour
    )
)
ORDER BY date DESC;
```

### Export CSV des zones à risque élevé

```sql
COPY (
    SELECT 
        date,
        hour,
        latitude,
        longitude,
        probability,
        atm_code
    FROM predictions
    WHERE date = CURRENT_DATE + INTERVAL 1 DAY
    AND probability >= 0.8
    ORDER BY hour, probability DESC
) TO 'high_risk_zones.csv' (HEADER, DELIMITER ',');
```

### Statistiques globales par date

```sql
SELECT 
    date,
    COUNT(*) as total_predictions,
    AVG(probability) as avg_risk,
    MIN(probability) as min_risk,
    MAX(probability) as max_risk,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY probability) as median_risk,
    COUNT(CASE WHEN probability >= 0.5 THEN 1 END) as moderate_risk,
    COUNT(CASE WHEN probability >= 0.8 THEN 1 END) as high_risk
FROM predictions
GROUP BY date
ORDER BY date DESC;
```

### Analyse temporelle - Risque par heure

```sql
SELECT 
    hour,
    AVG(probability) as avg_risk,
    COUNT(*) as nb_observations,
    COUNT(CASE WHEN probability >= 0.8 THEN 1 END) as high_risk_count
FROM predictions
GROUP BY hour
ORDER BY avg_risk DESC;
```

### Nettoyage - Supprimer les anciennes prédictions

```sql
-- Supprimer les prédictions de plus de 7 jours
DELETE FROM predictions 
WHERE date < CURRENT_DATE - INTERVAL 7 DAYS;

-- Vérifier l'espace libéré
VACUUM;
```

### Créer une vue pour les zones critiques

```sql
CREATE OR REPLACE VIEW zones_critiques AS
SELECT 
    date,
    hour,
    latitude,
    longitude,
    probability,
    CASE 
        WHEN probability >= 0.9 THEN 'Très élevé'
        WHEN probability >= 0.8 THEN 'Élevé'
        WHEN probability >= 0.6 THEN 'Moyen'
        ELSE 'Faible'
    END as niveau_risque
FROM predictions
WHERE probability >= 0.6;

-- Utilisation
SELECT * FROM zones_critiques WHERE date = CURRENT_DATE + INTERVAL 1 DAY;
```

### Export GeoJSON pour cartographie

```sql
INSTALL spatial;
LOAD spatial;

COPY (
    SELECT 
        date,
        hour,
        probability,
        ST_AsGeoJSON(ST_Point(longitude, latitude)) as geometry
    FROM predictions
    WHERE date = CURRENT_DATE + INTERVAL 1 DAY
    AND probability >= 0.7
) TO 'predictions.geojson' (FORMAT GDAL, DRIVER 'GeoJSON');
```
