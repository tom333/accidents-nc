# TODO Amélioration Qualité Prédictions

1. **[P1] Diversification des Algorithmes (Sortir des Arbres)**
    - Si vos trois modèles font des erreurs similaires, les combiner n'apportera rien. Il faut introduire des modèles qui "pensent" différemment.

    TabNet (Google) :

    Pourquoi ? C'est une architecture de Deep Learning conçue spécifiquement pour les données tabulaires. Elle utilise un mécanisme d'attention pour choisir les features.

    Valeur ajoutée : Il capture des relations non-linéaires complexes que les arbres peuvent manquer. Il est souvent un excellent candidat pour être moyenné avec un XGBoost.

    Réseaux de Neurones "Simples" (MLP avec Embeddings) :

    Pourquoi ? Un réseau dense (Dense Neural Network) avec des couches d'embedding pour vos variables catégorielles (ex: type de route, météo).

    Valeur ajoutée : Les réseaux de neurones gèrent mieux les interactions continues très fluides, là où les arbres "coupent" l'espace en rectangles.

    Modèles Linéaires Régularisés (Logistic Regression / Ridge / Lasso) :

    Pourquoi ? Cela peut sembler basique, mais une régression logistique bien calibrée capture les tendances linéaires globales.

    Valeur ajoutée : Essentiel si vous faites du Stacking (voir point 2).

2. **[P1] Techniques Avancées d'Ensembling : Le Stacking**

    - Plutôt que de choisir le meilleur modèle, ou de faire une moyenne simple, utilisez le Stacking.

    Le concept : Vous prenez les prédictions (probabilités) de CatBoost, XGBoost, LGBM et TabNet sur votre jeu de validation. Ces prédictions deviennent les "features" d'entrée d'un nouveau modèle (le méta-modèle).

    Le conseil d'expert : Utilisez une Régression Logistique ou un Ridge comme méta-modèle. Cela permet de pondérer intelligemment quel modèle a raison dans quel contexte, sans sur-apprendre.

3. **[P1]Gestion de la "Probabilité" (Calibration)**
    - Vous cherchez à prédire une probabilité d'accident. Or, les méthodes de Boosting (surtout XGBoost et LGBM) ont tendance à ne pas être bien calibrées (elles poussent les probabilités vers 0 ou 1).

Technique : CalibratedClassifierCV (Isotonic Regression ou Platt Scaling)

L'action : Après avoir entraîné votre modèle, passez-le dans un calibrateur.

Résultat : Une prédiction de 0.3 voudra vraiment dire "30% de risque", ce qui est crucial pour l'assurance ou la sécurité routière.

4. **[P1]Feature Engineering Spécifique "Route"**
Souvent, le gain de performance ne vient pas du modèle, mais de la donnée. Pour les accidents de la route, voici ce qui fait la différence :

Target Encoding (avec lissage) :

Très puissant pour les variables à haute cardinalité comme le "Code Postal" ou le "Nom de la Rue". CatBoost le fait nativement, mais aidez LGBM/XGBoost en le pré-calculant.

Features Géospatiales / Clustering :

N'utilisez pas juste Latitude/Longitude brutes.

Utilisez K-Means sur les coordonnées GPS pour créer une feature "Cluster de zone dangereuse".

Calculez la distance au centre-ville ou aux intersections majeures.

Interaction Features :

Créez manuellement des interactions logiques : Vitesse_Autorisée * Météo_Pluie ou Heure_Pointe * Type_Route.

5. **[P1]Gestion du Déséquilibre (Class Imbalance)**


Les accidents sont (heureusement) des événements rares par rapport aux trajets sans accident. Si vous avez 99% de "pas d'accident", vos modèles vont prédire "0" tout le temps et avoir 99% de précision (accuracy), mais être inutiles.

Focal Loss :

Au lieu de la LogLoss classique, implémentez la Focal Loss (disponible dans LGBM et XGBoost). Elle force le modèle à se concentrer sur les exemples "difficiles" (les accidents rares) plutôt que sur les cas faciles.

Scale Pos Weight :

Utilisez le paramètre scale_pos_weight (ou class_weights dans CatBoost) pour pénaliser plus fortement les erreurs sur la classe minoritaire (accident).

6. **[P1] Exploiter les variables BAAC manquantes**
   - Intégrer depuis `caracteristiques` et `lieux` : `lum`, `agg`, `int`, `plan`, `prof`, `surf`, `situ`, `infra`, `vma`, `catr`, `circ`, `nbv`, `vosp` (imputation/encodage des valeurs -1 à prévoir).
   - Ajouter des agrégations côté `vehicules` (ex. proportion 2RM, présence PL, manœuvre dominante `manv`, type de motorisation) pour chaque accident.
   - Mettre à jour la sélection de features et la Streamlit app pour consommer ces nouvelles colonnes.

7. **[P2] Mettre en place une validation plus robuste**
   - Ajouter une validation temporelle (train sur années historiques, test sur années récentes) pour éviter les fuites.
   - Tester une validation géographique (leave-one-commune/province-out) afin d'évaluer la généralisation spatiale.
   - Comparer recall/precision/F1 entre ces schémas et la validation aléatoire actuelle.

8. **[P2] Calibrer les probabilités et définir les seuils métier**
   - Utiliser `CalibratedClassifierCV` (Platt/isotonic) ou un jeu de validation dédié pour recalibrer `predict_proba`.
   - Tracer courbes de calibration et Precision-Recall pour choisir des seuils adaptés aux usages (grand public vs autorités).
   - Documenter les seuils retenus et exposer plusieurs profils d’affichage dans l’application (ex. « mode vigilance », « mode alerte »).

9. **[P3] Enrichir les features exogènes avancées**
   - Raffiner les features de densité (fenêtres spatiales/temporelles glissantes, séparation par type de route).
   - Ajouter des indicateurs d’infrastructure spécifiques (éclairage réel, virages serrés, tunnels, zones de travaux) issus d’OSM ou d’autres sources.
   - Intégrer des évènements ponctuels (week-ends prolongés, fêtes locales, météo temps réel) pour capter les contextes rares.
