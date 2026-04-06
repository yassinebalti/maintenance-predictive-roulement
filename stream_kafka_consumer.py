"""
Kafka Consumer pour Stream Processing
Remplace update.py : Lit flux capteurs, traite et alimente pipeline.
Installe : pip install kafka-python
Lancer Kafka d'abord : zookeeper-server-start, kafka-server-start
Créer topic : kafka-topics --create --topic capteurs-data --bootstrap-server localhost:9092
"""

from kafka import KafkaConsumer
import json
import pandas as pd
import subprocess
import os
from datetime import datetime

# Config Kafka
BOOTSTRAP_SERVER = 'localhost:9092'
TOPIC = 'capteurs-data'

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVER,
    auto_offset_reset='earliest',  # Commence du début si nouveau
    enable_auto_commit=True,
    group_id='maintenance-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Fichier temp pour nouvelles données
TEMP_CSV = 'data/new_data.csv'

def process_new_data(new_measure):
    # Ajoute à CSV temp
    df_new = pd.DataFrame([new_measure])
    if os.path.exists(TEMP_CSV):
        df_new.to_csv(TEMP_CSV, mode='a', header=False, index=False)
    else:
        df_new.to_csv(TEMP_CSV, index=False)

    # Si assez de données (ex: 20), lance pipeline
    if len(pd.read_csv(TEMP_CSV)) >= 20:
        print(f"[{datetime.now()}] Nouvelles données : Lancement pipeline")
        # Fusionne avec données existantes
        df_old = pd.read_csv('data/01_raw_motor.csv')
        df_combined = pd.concat([df_old, pd.read_csv(TEMP_CSV)])
        df_combined.to_csv('data/01_raw_motor.csv', index=False)
        # Lance pipeline
        subprocess.run(['python', 'main_pipeline.py', '--from-step', '2'])
        # Vide temp
        open(TEMP_CSV, 'w').close()

print("Kafka Consumer actif...")

for message in consumer:
    data = message.value
    print(f"Reçu : {data}")
    process_new_data(data)

# Exemple envoi test (producteur) :
# from kafka import KafkaProducer
# producer = KafkaProducer(bootstrap_servers=BOOTSTRAP_SERVER, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
# producer.send(TOPIC, {'motor_id': 1, 'timestamp': str(datetime.now()), 'temperature': 35.0, 'vibration': 0.7})