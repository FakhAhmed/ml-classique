# Système MLOps de Prédiction d'Attrition (Churn) sur GCP

Une architecture Machine Learning de bout en bout, de l'entraînement du modèle XGBoost jusqu'à son déploiement en production. Ce projet démontre la mise en place d'un pipeline MLOps complet, automatisé, découplé et sécurisé sur l'écosystème Google Cloud.

- **Lien de l'API :** [API de Prédiction Churn](https://api-churn-ml-306641727979.europe-west9.run.app/docs) *(Endpoint `/predict` via POST)*

## 🚀 Fonctionnalités Clés
- **Découplage Code/Modèle (Dynamic Fetching) :** Le modèle n'est plus "hardcodé" dans l'image Docker. L'API télécharge dynamiquement le fichier `.bst` au démarrage depuis un Bucket Cloud Storage sécurisé.
- **Gouvernance des Modèles (Vertex AI) :** Intégration du Model Registry de Google. L'API interroge le catalogue officiel de l'entreprise pour toujours récupérer et servir la version de production certifiée.
- **CI/CD Native (GitOps) :** Déploiement entièrement automatisé via **Google Cloud Build**. Chaque `git push` sur la branche `main` déclenche le linting, le build Docker, le push sur Artifact Registry et le déploiement Serverless sans intervention humaine.
- **Sécurité "Least Privilege" (IAM) :** L'API s'exécute sous un compte de service dédié, doté uniquement des permissions strictes de lecture (Object Viewer / AI Platform Viewer), garantissant un cloisonnement total en cas de faille.
- **API Robuste & Résiliente :** API **FastAPI** avec validation Pydantic. Implémentation d'un "gilet de sauvetage" (Data Imputation) : le code aligne dynamiquement les requêtes entrantes sur les 30 features requises par XGBoost pour éviter tout crash serveur.

## 🏗️ Architecture Technique
- **Machine Learning :** XGBoost, Scikit-Learn, Pandas
- **Service Web :** FastAPI, Uvicorn, Pydantic
- **Automatisation (CI/CD) :** Google Cloud Build
- **Registre de Conteneurs :** Google Artifact Registry
- **Déploiement Serverless :** Google Cloud Run
- **Stockage Artefacts :** Google Cloud Storage (Buckets)
- **Registre MLOps :** Google Vertex AI Model Registry
- **Sécurité / Identité :** Google Cloud IAM (Service Accounts)

## 📁 Structure du Projet
- **`api/`** : Contient le code source de l'API REST (`main.py`). Il gère la validation JSON, l'interrogation de l'API Vertex, le téléchargement Cloud Storage et l'inférence.
- **`cloudbuild.yaml`** : Le cerveau de l'automatisation. Fichier d'orchestration définissant les étapes du pipeline CI/CD (Build, Push, Deploy, Logging).
- **`data/`** : Espace de stockage des jeux de données d'entraînement (données brutes `telco_churn.csv` et processées).
- **`notebooks/`** : Laboratoire de recherche (`01_exploration_churn.ipynb`) contenant l'analyse exploratoire des données (EDA) et l'entraînement mathématique du modèle initial.
- **`Dockerfile`** : Recette de conteneurisation assurant la portabilité absolue de l'application entre les environnements de développement et de production.
- **`requirements.txt`** : Liste exhaustive des dépendances Python, incluant les SDK Google Cloud (`google-cloud-storage`, `google-cloud-aiplatform`).

## 💡 Pourquoi ce projet ?
Ce PoC illustre le fossé entre le travail théorique en Data Science et l'industrialisation (MLOps). Il prouve la capacité à transformer un modèle mathématique isolé en un microservice d'entreprise : robuste aux erreurs de données, déployé via des pipelines automatisés (GitOps), et sécurisé par des stratégies IAM strictes. C'est l'essence même de l'Ingénierie Data & IA moderne.