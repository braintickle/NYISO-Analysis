FROM public.ecr.aws/lambda/python:3.11

WORKDIR ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.9 \
    requests==2.31.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    xgboost==2.0.3 \
    scikit-learn==1.4.2 \
    joblib==1.4.2 \
    holidays==0.46 \
    pulp==2.8.0 \
    python-dotenv==1.0.1

COPY lambda/ ./
COPY models/ ./models/

CMD ["handler.lambda_handler"]
