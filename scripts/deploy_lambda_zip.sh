#!/usr/bin/env bash
# deploy_lambda_zip.sh — Deploy NYISO Lambda using zip + Layer (no Docker required)
#
# Prerequisites:
#   1. Run this script first: pip install --platform manylinux2014_x86_64 ...
#      (already done by Claude — packages are in lambda_layer/python/)
#   2. AWS CLI configured with nyiso-deploy IAM user
#   3. nyiso-deploy has iam/nyiso-deploy-policy.json attached
#   4. models/*.joblib exist (run scripts/export_models.py first — already done)
#
# Run from repo root:
#   bash scripts/deploy_lambda_zip.sh

set -euo pipefail

REGION="us-east-1"
ACCOUNT="358592704128"
FUNCTION_NAME="nyiso-ingestor"
ROLE_NAME="nyiso-lambda-role"
LAYER_NAME="nyiso-deps"
IMAGE_TAG="$(git rev-parse --short HEAD)"

set -a; source .env; set +a
if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "ERROR: DATABASE_URL not set in .env" >&2; exit 1
fi

echo "=== Step 1: Create Layer zip (lambda_layer/python/ → nyiso-deps-layer.zip) ==="
if [[ ! -d "lambda_layer/python" ]]; then
  echo "ERROR: lambda_layer/python/ not found." >&2
  echo "Run the pip install command first — see instructions in this script header." >&2
  exit 1
fi
cd lambda_layer
zip -r ../nyiso-deps-layer.zip python/ -q
cd ..
LAYER_SIZE=$(du -sh nyiso-deps-layer.zip | cut -f1)
echo "  Layer zip: ${LAYER_SIZE}  (nyiso-deps-layer.zip)"

echo ""
echo "=== Step 2: Create function zip (handler code + models) ==="
rm -f nyiso-function.zip
zip -j nyiso-function.zip lambda/handler.py lambda/ingest.py lambda/features.py lambda/inference.py lambda/bess_dispatch.py -q
zip -r nyiso-function.zip models/ -q
FUNC_SIZE=$(du -sh nyiso-function.zip | cut -f1)
echo "  Function zip: ${FUNC_SIZE}  (nyiso-function.zip)"

echo ""
echo "=== Step 3: Create IAM execution role (idempotent) ==="
aws iam get-role --role-name "${ROLE_NAME}" > /dev/null 2>&1 && \
  echo "  Role already exists — skipping." || \
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{
        "Effect":"Allow",
        "Principal":{"Service":"lambda.amazonaws.com"},
        "Action":"sts:AssumeRole"
      }]
    }' \
    --description "Execution role for NYISO Lambda ingestor"

aws iam attach-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole \
  2>/dev/null || true

ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${ROLE_NAME}"
echo "  Role ARN: ${ROLE_ARN}"
sleep 12   # IAM propagation

echo ""
echo "=== Step 4: Publish Lambda Layer ==="
LAYER_ARN=$(aws lambda publish-layer-version \
  --layer-name "${LAYER_NAME}" \
  --description "NYISO ML dependencies (pandas, xgboost, psycopg2, pulp)" \
  --zip-file fileb://nyiso-deps-layer.zip \
  --compatible-runtimes python3.11 \
  --region "${REGION}" \
  --query 'LayerVersionArn' --output text)
echo "  Layer ARN: ${LAYER_ARN}"

echo ""
echo "=== Step 5: Create or update Lambda function ==="
if aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${REGION}" > /dev/null 2>&1; then
  echo "  Function exists — updating code + config..."
  aws lambda update-function-code \
    --function-name "${FUNCTION_NAME}" \
    --zip-file fileb://nyiso-function.zip \
    --region "${REGION}"
  sleep 5
  aws lambda update-function-configuration \
    --function-name "${FUNCTION_NAME}" \
    --layers "${LAYER_ARN}" \
    --environment "Variables={DATABASE_URL=${DATABASE_URL},MODEL_VERSION=${IMAGE_TAG}}" \
    --region "${REGION}"
else
  echo "  Creating function..."
  aws lambda create-function \
    --function-name "${FUNCTION_NAME}" \
    --runtime python3.11 \
    --role "${ROLE_ARN}" \
    --handler handler.lambda_handler \
    --zip-file fileb://nyiso-function.zip \
    --layers "${LAYER_ARN}" \
    --timeout 300 \
    --memory-size 1024 \
    --environment "Variables={DATABASE_URL=${DATABASE_URL},MODEL_VERSION=${IMAGE_TAG}}" \
    --region "${REGION}"
fi

FUNC_ARN="arn:aws:lambda:${REGION}:${ACCOUNT}:function:${FUNCTION_NAME}"

echo ""
echo "=== Step 6: EventBridge rules ==="

# Rule 1: load + fuel — every 5 minutes
aws events put-rule --name "nyiso-ingest-load-fuel" \
  --schedule-expression "rate(5 minutes)" --state ENABLED \
  --description "NYISO load_actual + fuel_mix ingestion" --region "${REGION}"
aws lambda add-permission --function-name "${FUNCTION_NAME}" \
  --statement-id "allow-events-load-fuel" --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT}:rule/nyiso-ingest-load-fuel" \
  --region "${REGION}" 2>/dev/null || true
aws events put-targets --rule "nyiso-ingest-load-fuel" --region "${REGION}" \
  --targets "Id=1,Arn=${FUNC_ARN},Input={\"task\":\"ingest_load_fuel\"}"

# Rule 2: LMP + forecasting — every hour
aws events put-rule --name "nyiso-ingest-lmp" \
  --schedule-expression "rate(1 hour)" --state ENABLED \
  --description "NYISO LMP ingestion + XGBoost forecast update" --region "${REGION}"
aws lambda add-permission --function-name "${FUNCTION_NAME}" \
  --statement-id "allow-events-lmp" --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT}:rule/nyiso-ingest-lmp" \
  --region "${REGION}" 2>/dev/null || true
aws events put-targets --rule "nyiso-ingest-lmp" --region "${REGION}" \
  --targets "Id=1,Arn=${FUNC_ARN},Input={\"task\":\"ingest_lmp\"}"

# Rule 3: BESS dispatch — daily at 1pm UTC (9am ET)
aws events put-rule --name "nyiso-bess-dispatch" \
  --schedule-expression "cron(0 13 * * ? *)" --state ENABLED \
  --description "NYISO BESS daily dispatch optimization" --region "${REGION}"
aws lambda add-permission --function-name "${FUNCTION_NAME}" \
  --statement-id "allow-events-bess" --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT}:rule/nyiso-bess-dispatch" \
  --region "${REGION}" 2>/dev/null || true
aws events put-targets --rule "nyiso-bess-dispatch" --region "${REGION}" \
  --targets "Id=1,Arn=${FUNC_ARN},Input={\"task\":\"bess_dispatch\"}"

echo ""
echo "==================================================================="
echo "Lambda deployment complete!"
echo "  Function:  ${FUNCTION_NAME}  (${FUNC_SIZE})"
echo "  Layer:     ${LAYER_NAME}  (${LAYER_SIZE})"
echo "  Image tag: ${IMAGE_TAG}"
echo ""
echo "EventBridge rules:"
echo "  nyiso-ingest-load-fuel  → every 5 minutes"
echo "  nyiso-ingest-lmp        → every hour"
echo "  nyiso-bess-dispatch     → daily 1pm UTC / 9am ET"
echo ""
echo "Smoke test (run now):"
echo "  aws lambda invoke --function-name ${FUNCTION_NAME} \\"
echo "    --payload '{\"task\":\"ingest_load_fuel\"}' \\"
echo "    --cli-binary-format raw-in-base64-out /tmp/out.json \\"
echo "    --region ${REGION} && cat /tmp/out.json"
echo "==================================================================="
