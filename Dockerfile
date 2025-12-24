FROM python:3.11-slim

WORKDIR /code

# 1) Copier requirements.txt
COPY requirements.txt .

# 2) Mettre pip à jour (souvent nécessaire avec psycopg)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 3) Copier le code
COPY . .

# 4) Commande par défaut (tu peux l’override dans docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
