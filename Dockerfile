FROM jupyter/scipy-notebook

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN mkdir /home/jovyan/model
ENV MODEL_DIR=/home/jovyan/model
ENV MODEL_FILE=mlp_final.joblib
ENV METADATA_FILE=metadata.json

# Copy the data directory into the Docker image
COPY data /home/jovyan/data

# List the contents of the data directory to verify
RUN ls -l /home/jovyan/data

COPY train.py /home/jovyan/train.py
COPY api.py /home/jovyan/api.py

RUN python3 /home/jovyan/train.py

# Expose the port
EXPOSE 5000

# Command to run web server
CMD ["python3", "/home/jovyan/api.py"]