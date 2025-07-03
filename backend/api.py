import os
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv
import logging
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status

# Charger les variables d'environnement
load_dotenv()

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API de Détection de Mélanome",
    description="API pour la détection de mélanome à partir d'images de lésions cutanées",
    version="1.0.0",
    contact={
        "name": "Support",
        "email": "support@melanoma-detection.com"
    },
    license_info={
        "name": "Proprietary"
    }
)


# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connexion à MongoDB
def get_db():
    """Établit une connexion à la base de données MongoDB."""
    try:
        client = MongoClient(os.getenv("MONGODB_URI"))
        db = client[os.getenv("MONGODB_DB", "melanoma")]
        fs = gridfs.GridFS(db)
        
        # S'assurer que la collection predictions existe
        if "predictions" not in db.list_collection_names():
            db.create_collection("predictions")
            
        return db, fs
    except Exception as e:
        logger.error(f"Erreur de connexion à MongoDB: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service de base de données indisponible"
        )
    
# Chargement du modèle
def load_model():
    """Charge le modèle de détection de mélanome."""
    try:
        # Chemins possibles pour le modèle
        model_paths = [
            "/app/models/cancer_detection_model.keras",  # Chemin dans le conteneur
            "./models/cancer_detection_model.keras"      # Chemin local pour le développement
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                logger.info(f"Modèle trouvé à l'emplacement: {path}")
                break
                
        if not model_path:
            error_msg = f"Fichier du modèle introuvable. Vérifiez qu'il se trouve dans l'un de ces emplacements: {', '.join(model_paths)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Charger le modèle complet
        model = tf.keras.models.load_model(model_path)
        logger.info("Modèle chargé avec succès")
        return model
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du chargement du modèle: {str(e)}"
        )

# Initialisation du modèle
model = load_model()

# Fonction de prétraitement de l'image
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Prétraite l'image pour la prédiction.
    
    Args:
        image: Image PIL à prétraiter
        
    Returns:
        np.ndarray: Image prétraitée sous forme de tableau numpy
    """
    try:
        # Conversion en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Redimensionnement et normalisation
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        
        # Vérification des dimensions
        if len(image_array.shape) == 2:  # Image en niveaux de gris
            image_array = np.stack((image_array,) * 3, axis=-1)
        elif image_array.shape[2] == 4:  # Image avec canal alpha
            image_array = image_array[..., :3]
            
        return np.expand_dims(image_array, axis=0)  # Ajout de la dimension du batch
        
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement de l'image: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors du traitement de l'image: {str(e)}"
        )

# Endpoint de santé
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy"}

# Endpoints de l'API
@app.get("/", tags=["Health Check"])
async def root():
    """
    Vérifie que l'API est opérationnelle.
    """
    return {
        "status": "ok",
        "message": "API de détection de mélanome opérationnelle",
        "version": "1.0.0"
    }

@app.post("/predict", tags=["Prédiction"])
@app.post("/predict", tags=["Prédiction"])
async def predict(file: UploadFile, db_data: tuple = Depends(get_db)) -> Dict[str, Any]:
    db, fs = db_data
    
    # Ajout de logs
    logger.info(f"Reçu une requête de prédiction pour le fichier: {file.filename}")
    logger.info(f"Type de contenu: {file.content_type}")
    
    # Vérification du type de fichier
    if not file.content_type.startswith('image/'):
        error_msg = f"Type de fichier non supporté: {file.content_type}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    
    try:    
        # Lecture de l'image
        contents = await file.read()
        logger.info(f"Taille du fichier: {len(contents)} octets")
        
        # Vérification de la taille du fichier (max 16MB)
        if len(contents) > 16 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="La taille du fichier dépasse la limite de 16MB"
            )
            
        # Chargement de l'image
        image = Image.open(io.BytesIO(contents))
        logger.info("Image chargée avec succès")
        
        # Prétraitement
        processed_image = preprocess_image(image)
        logger.info("Image prétraitée avec succès")
        
        # Prédiction
        prediction = model.predict(processed_image, verbose=0)
        probability = float(prediction[0][0])
        result = "malin" if probability > 0.5 else "bénin"
        logger.info(f"Prédiction effectuée: {result} (confiance: {probability:.2f})")
        
        # Préparation de la réponse
        response = {
            "prediction_id": str(uuid.uuid4()),
            "result": result,
            "confidence": probability,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Sauvegarde dans MongoDB (optionnel)
            prediction_data = {
                **response,
                "filename": file.filename,
                "content_type": file.content_type,
                "upload_date": datetime.utcnow(),
                "size": len(contents)
            }
            
            # Sauvegarde dans GridFS
            file_id = fs.put(
                contents,
                filename=file.filename,
                content_type=file.content_type,
                metadata=prediction_data
            )
            
            # Sauvegarde des métadonnées
            db.predictions.insert_one({
                **prediction_data,
                "gridfs_file_id": file_id
            })
            logger.info("Prédiction sauvegardée dans MongoDB")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde dans MongoDB: {str(e)}")
            # On continue même en cas d'échec de sauvegarde
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Une erreur est survenue lors du traitement de la requête: {str(e)}"
        )

@app.get("/predictions/{prediction_id}", tags=["Historique"])
async def get_prediction(prediction_id: str, db_data: tuple = Depends(get_db)) -> Dict[str, Any]:
    """
    Récupère les détails d'une prédiction précédente.
    
    Args:
        prediction_id: ID de la prédiction à récupérer
        
    Returns:
        Détails de la prédiction
    """
    db, fs = db_data
    
    try:
        # Récupération des métadonnées
        prediction = db.predictions.find_one({"file_id": prediction_id})
        
        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prédiction non trouvée"
            )
        
        # Conversion de l'ObjectId en chaîne pour la sérialisation JSON
        prediction["_id"] = str(prediction["_id"])
        
        # Vérification de la disponibilité de l'image
        if "gridfs_file_id" in prediction:
            try:
                grid_fs_file = fs.get(prediction["gridfs_file_id"])
                prediction["image_available"] = grid_fs_file.length > 0
            except:
                prediction["image_available"] = False
        
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la prédiction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la récupération des données"
        )
