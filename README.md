# ML-Covid-Predictions

This repository contains a machine learning model for flagging high risk patients already diagnosed with Covid-19. Given a large number of COVID-19 patients, the model can be used to decide which patients require urgent care, for example by a hospital, to ensure that they receive emergency care to minimise potential complications. It is trained and fine-tuned on data public domain data provided by the Mexican government, available at [Kaggle](https://www.kaggle.com/datasets/meirnizri/covid19-dataset/). The model has over 95% recall - it will detect and flag over 95% high risk patients, but it will also flag many lower risk patients as high risk. You can the model run using Docker.

## Prerequisites

- Docker installed on your machine

## Steps to Run the Model

1. **Navigate to the project directory:**

    ```sh
    cd /yourpath/ML-Covid-Predictions
    ```

2. **Build the Docker image:**

    ```sh
    docker build -t docker-api -f Dockerfile .
    ```

3. **Run the Docker container:**

    ```sh
    docker run -it -p 5001:5000 docker-api python3 api.py
    ```

4. **Make a prediction request:**

    ```sh
    curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d '{
        "USMER": 1,
        "SEX": "Male",
        "PATIENT_TYPE": "Home", 
        "INTUBED": 0,
        "PNEUMONIA": 0,
        "AGE": 40,
        "PREGNANT": 0,
        "DIABETES": 0,
        "COPD": 0,
        "ASTHMA": 0,
        "INMSUPR": 0,
        "HIPERTENSION": 0,
        "OTHER_DISEASE": 0,
        "CARDIOVASCULAR": 0,
        "OBESITY": 0,
        "RENAL_CHRONIC": 0,
        "TOBACCO": 0,
        "CLASIFFICATION_FINAL": 1,
        "ICU": 0
    }'
    ```

## Notes

- Ensure that the Docker daemon is running before executing the Docker commands.
- The API will be accessible at `http://localhost:5001` once the Docker container is running. If configured correctly, the message "Welcome to the Flask API!" will appear on screen. 

## Troubleshooting

- Ensure that your Docker installation is up to date.
- Check the logs of the Docker container for any errors by running:

    ```sh
    docker logs <container_id>
    ```

Replace `<container_id>` with the actual container ID obtained from running containers:

    ```sh
    docker ps
    ```
