FROM python:3.10-slim
EXPOSE 8523
WORKDIR /usr/src/app
COPY requirements.txt ./
COPY app.py ./
RUN pip install -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["streamlit", "run", "--server.enableCORS", "false", "--server.port", "8523", "--server.enableXsrfProtection", "false", "--server.address", "0.0.0.0", "app.py"] 