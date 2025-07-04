# 🏥 Détection de Mélanome par Deep Learning

Application de détection de mélanome utilisant le deep learning pour classifier les lésions cutanées à partir d'images médicales.

## 🚀 Fonctionnalités Principales

- **Classification automatique** des lésions cutanées (bénignes/malignes)
- **Interface utilisateur intuitive** pour le téléchargement et l'analyse d'images
- **Historique des prédictions** avec stockage sécurisé
- **API RESTful** pour une intégration facile
- **Déploiement conteneurisé** avec Docker

## 🗂️ Architecture du Projet
```bash
Projet-Deep-Learning/
│
├── .github/
│   └── workflows/
│       └── deploy.yml          # Configuration CI/CD
│
├── backend/                    # API FastAPI
│   ├── api.py                 # Points d'entrée de l'API
│   ├── requirements.api.txt    # Dépendances Python
│   └── Dockerfile             # Configuration Docker
│
├── frontend/                  # Interface Streamlit
│   ├── frontend.py            # Interface utilisateur
│   ├── requirements.frontend.txt
│   └── Dockerfile
│
├── models/                    # Modèles de deep learning
│   └── cancer_detection_model.keras
│
├── Notebook/                  # Notebooks Jupyter
│   └── model_training.ipynb   # Entraînement du modèle
│
├── docker-compose.yml         # Configuration multi-conteneurs
├── .env.example              # Exemple de variables d'environnement
└── README.md                 # Ce fichier
```

## 🛠️ Technologies Utilisées

### Backend (FastAPI)
- **Framework** : FastAPI
- **Base de données** : MongoDB avec GridFS pour le stockage des images
- **Traitement d'images** : OpenCV, TensorFlow/Keras
- **Authentification** : JWT
- **Tests** : Pytest

### Frontend (Streamlit)
- **Framework** : Streamlit
- **Interface** : Composants personnalisés avec CSS
- **Gestion d'état** : Session State
- **Visualisation** : Matplotlib, Plotly

### Infrastructure
- **Conteneurisation** : Docker, Docker Compose
- **CI/CD** : GitHub Actions
- **Hébergement** : AWS EC2
- **Gestion des secrets** : GitHub Secrets

## 🚀 Démarrage Rapide

### Prérequis
- Docker 20.10+
- Docker Compose 2.0+
- Git

### Installation
```bash
# Cloner le dépôt
git clone [https://github.com/votre-utilisateur/Deep-Learning-Detection-Melanome.git](https://github.com/votre-utilisateur/Deep-Learning-Detection-Melanome.git)
cd Deep-Learning-Detection-Melanome

# Lancer les services en localhost avec docker-compose
docker-compose up --build

# 🧠 Modèle de Deep Learning

## 🏗️ Architecture du Modèle

### Architecture Principale
- **Type de modèle** : Réseau de neurones convolutif (CNN)
- **Architecture de base** : MobileNetV2 avec transfer learning
- **Paramètres** :
  - Total : 2,340,033 paramètres (8.93 MB)
  - Entraînables : 82,049 paramètres (320.50 KB)
  - Non-entraînables : 2,257,984 paramètres (8.61 MB)

### 🔧 Configuration d'Entraînement
- **Optimiseur** : Adam
- **Fonction de perte** : Binary Cross-Entropy
- **Métriques** : Accuracy, AUC
- **Early Stopping** : Patience de 3 époques
- **Nombre d'époques** : 5 (avec possibilité d'augmenter)
- **Taille du batch** : Défini dans le générateur de données

## 📊 Performance du Modèle

### Métriques Principales
- **AUC sur validation** : 0.8479
- **Perte sur validation** : 0.4910
- **Précision globale** : 76.07%

### Rapport de Classification Détailé
| Métrique | Classe 0 | Classe 1 | Support |
|----------|----------|----------|---------|
| Précision | 76.99% | 75.21% | 117 |
| Rappel | 74.36% | 77.78% | 117 |
| F1-Score | 75.65% | 76.47% | 117 |

## 📈 Analyse des Résultats

### Courbes d'Apprentissage
1. **AUC (Area Under Curve)**
   - L'AUC d'entraînement augmente régulièrement
   - L'AUC de validation se stabilise autour de 0.85
   - Bonne capacité de généralisation

2. **Perte (Loss)**
   - Diminution constante de la perte d'entraînement
   - Stabilité de la perte de validation
   - Aucun signe de surapprentissage

### Points Forts
- **Bonne généralisation** : Performance similaire sur les ensembles d'entraînement et de validation
- **Équilibre** : Métriques équilibrées entre les deux classes
- **Stabilité** : Apprentissage stable sans sur-ajustement

### Domaines d'Amélioration
- **Précision** : Possibilité d'augmenter la taille du jeu de données
- **Régularisation** : Pourrait être optimisée pour améliorer les performances
- **Architecture** : Essayer d'autres modèles de base pour comparer les performances

## 🚀 Utilisation

Le modèle est prêt à être utilisé pour la classification des lésions cutanées via l'API FastAPI. Il prend en entrée des images de lésions dermatologiques et retourne une prédiction binaire (bénin/malin) accompagnée d'un score de confiance.

---

# 🚀 Déploiement Automatisé

## 🌐 Accès au Déploiement

L'application est déployée et accessible à l'adresse :  
🔗 [http://54.170.1.218:8501/](http://54.170.1.218:8501/)

### Points d'Accès
- **Interface Utilisateur** : http://54.170.1.218:8501
- **API Documentation** : http://54.170.1.218:8000/docs
- **Health Check** : http://54.170.1.218:8000/health


## 🔄 Workflow CI/CD

Le déploiement est entièrement automatisé via GitHub Actions :

### Déclencheurs
- Push sur la branche `main`
- Déclenchement manuel depuis l'interface GitHub

### Étapes du Déploiement

1. **Vérification du code**
   - Lint du code Python
   - Exécution des tests unitaires

2. **Construction des images Docker**
   - Backend (FastAPI)
   - Frontend (Streamlit)
   - Base de données MongoDB

3. **Déploiement sur EC2**
   - Connexion SSH sécurisée
   - Arrêt des conteneurs existants
   - Nettoyage des ressources inutilisées
   - Démarrage des nouveaux conteneurs

4. **Vérification du déploiement**
   - Tests de santé des services
   - Vérification des logs

## 🔧 Configuration Requise

### Variables d'Environnement (GitHub Secrets)
- `AWS_SSH_KEY` : Clé privée pour la connexion SSH
- `AWS_HOST` : Adresse IP de l'instance EC2
- `MONGODB_URI` : URI de connexion à MongoDB
- `API_SECRET` : Clé secrète pour l'API

### Ressources Serveur
- **CPU** : 2 cœurs minimum
- **RAM** : 4 Go minimum
- **Stockage** : 20 Go d'espace disque
- **Ports ouverts** : 22 (SSH), 8000 (API), 8501 (Frontend)

## 🔄 Rollback Automatique

En cas d'échec du déploiement :
1. Notification par email
2. Rollback vers la version précédente
3. Conservation des logs pour débogage

## 📊 Monitoring

- **Logs** : Centralisés avec Docker
- **Métriques** : CPU, mémoire, espace disque
- **Alertes** : En cas d'indisponibilité

## 🔒 Sécurité

- Connexion SSH par clé uniquement
- Renouvellement automatique des certificats SSL
- Mises à jour de sécurité automatiques
- Backup quotidien de la base de données

---

## 👨‍💻 À Propos de Moi

Je suis **Data Engineer & Data Scientist Junior** passionné par le machine learning et le développement d'applications IA. Ce projet illustre mes compétences en deep learning et en déploiement d'applications.

## 📧 Contact

- **Email** : [bousmane733@gmail.com](mailto:bousmane733@gmail.com)
- **GitHub** : [https://github.com/Ousmane-BA100](https://github.com/Ousmane-BA100)

## 📄 Licence

Distribué sous la licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

---

✨ **Merci d'avoir consulté ce projet !** N'hésitez pas à me contacter pour toute opportunité ou collaboration.