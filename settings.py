import os
import firebase_admin
from firebase_admin import credentials, firestore, auth, storage

CREDENTIAL_FILE = credentials.Certificate('fire_detection.json')
app = firebase_admin.initialize_app(CREDENTIAL_FILE, {'storageBucket': 'fireandsmoke-f33cc.appspot.com'})
FIRESTORE_DB = firestore.client()
FIREBASE_AUTH = auth
FIREBASE_BUCKET = storage.bucket()
MEDIA_ROOT = "media"