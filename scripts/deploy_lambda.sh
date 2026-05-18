#!/usr/bin/env bash
# deploy_lambda.sh — Build, push, and configure the NYISO Lambda function
#
# Prerequisites:
#   1. Docker Desktop running
#   2. AWS CLI configured (aws sts get-caller-identity should succeed)
#   3. nyiso-deploy IAM user has iam/nyiso-deploy-policy.json attached
#   4. DATABASE_URL set in .env (already done)
#   5. models/*.joblib exist (run scripts/export_models.py first)
#
# Run from repo root:
#   bash scripts/deploy_lambda.sh

set -euo pipefail

REGION="us-east-1"
ACCOUNT="358592704128"
REPO_NAME="nyiso-lambda"
FUNCTION_NAME="nyiso-ingestor"
ROLE_NAME="nyiso-lambda-role"
IMAGE_TAG="$(git rev-parse --short HEAD)"
ECR_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}"

# Pull DATABASE_URL from .env
set -a; source .env; set +a
if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "ERROR: DATABASE_URL not set in .env" >&2
  exit 1
fi

echo "=== Step 1: Create ECR repository (idempotent) ==="
aws ecr create-repository \
  --repository-name "${REPO_NAME}" \
  --region "${REGION}" \
  --image-scanning-configuration scanOnPush=true \
  2>/dev/null || echo "  Repository already exists — skipping."

echo ""
echo "=== Step 2: Docker login to ECR ==="
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

echo ""
echo "=== Step 3: Build Lambda container image ==="
docker build \
  --platform linux/amd64 \
  -f lambda/Dockerfile \
  -t "${ECR_URI}:${IMAGE_TAG}" \
  -t "${ECR_URI}:latest" \
  .

echo ""
echo "=== Step 4: Push to ECR ==="
docker push "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:latest"

echo ""
echo "=== Step 5: Create IAM execution role (idempotent) ==="
aws iam get-role --role-name "${ROLE_NAME}" > /dev/null 2>&1 || \
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

# Wait a few seconds for IAM to propagate
sleep 10

echo ""
echo "=== Step 6: Create or update Lambda function ==="
if aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${REGION}" > /dev/null 2>&1; then
  echo "  Function exists — updating code..."
  aws lambda update-function-code \
    --function-name "${FUNCTION_NAME}" \
    --image-uri "${ECR_URI}:${IMAGE_TAG}" \
    --region "${REGION}"
else
  echo "  Creating function..."
  aws lambda create-function \
    --function-name "${FUNCTION_NAME}" \
    --package-type Image \
    --code ImageUri="${ECR_URI}:${IMAGE_TAG}" \
    --role "${ROLE_ARN}" \
    --timeout 300 \
    --memory-size 1024 \
    --environment "Variables={DATABASE_URL=${DATABASE_URL},MODEL_VERSION=${IMAGE_TAG}}" \
    --region "${REGION}"
fi

echo ""
echo "=== Step 7: EventBridge rules ==="

FUNC_ARN="arn:aws:lambda:${REGION}:${ACCOUNT}:function:${FUNCTION_NAME}"

# Rule 1: Load + fuel — every 5 minutes
aws events put-rule \
  --name "nyiso-ingest-load-fuel" \
  --schedule-expression "rate(5 minutes)" \
  --state ENABLED \
  --description "NYISO load_actual + fuel_mix ingestion" \
  --region "${REGION}"

aws lambda add-permission \
  --function-name "${FUNCTION_NAME}" \
  --statement-id "allow-events-load-fuel" \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT}:rule/nyiso-ingest-load-fuel" \
  --region "${REGION}" 2>/dev/null || true

aws events put-targets \
  --rule "nyiso-ingest-load-fuel" \
  --targets "Id=1,Arn=${FUNC_ARN},Input={\"task\":\"ingest_load_fuel\"}" \
  --region "${REGION}"

# Rule 2: LMP + forecasting — every hour
aws events put-rule \
  --name "nyiso-ingest-lmp" \
  --schedule-expression "rate(1 hour)" \
  --state ENABLED \
  --description "NYISO LMP ingestion + XGBoost forecast update" \
  --region "${REGION}"

aws lambda add-permission \
  --function-name "${FUNCTION_NAME}" \
  --statement-id "allow-events-lmp" \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT}:rule/nyiso-ingest-lmp" \
  --region "${REGION}" 2>/dev/null || true

aws events put-targets \
  --rule "nyiso-ingest-lmp" \
  --targets "Id=1,Arn=${FUNC_ARN},Input={\"task\":\"ingest_lmp\"}" \
  --region "${REGION}"

# Rule 3: BESS dispatch — daily at 1pm UTC (9am ET, after DA market clears)
aws events put-rule \
  --name "nyiso-bess-dispatch" \
  --schedule-expression "cron(0 13 * * ? *)" \
  --state ENABLED \
  --description "NYISO BESS daily dispatch optimization" \
  --region "${REGION}"

aws lambda add-permission \
  --function-name "${FUNCTION_NAME}" \
  --statement-id "allow-events-bess" \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT}:rule/nyiso-bess-dispatch" \
  --region "${REGION}" 2>/dev/null || true

aws events put-targets \
  --rule "nyiso-bess-dispatch" \
  --targets "Id=1,Arn=${FUNC_ARN},Input={\"task\":\"bess_dispatch\"}" \
  --region "${REGION}"

echo ""
echo "==================================================================="
echo "Lambda deployment complete!"
echo "  Function:   ${FUNCTION_NAME}"
echo "  Image:      ${ECR_URI}:${IMAGE_TAG}"
echo "  Role:       ${ROLE_ARN}"
echo ""
echo "EventBridge rules:"
echo "  nyiso-ingest-load-fuel  → rate(5 minutes)"
echo "  nyiso-ingest-lmp        → rate(1 hour)"
echo "  nyiso-bess-dispatch     → cron(0 13 * * ? *)  [1pm UTC / 9am ET]"
echo ""
echo "Test with:"
echo "  aws lambda invoke --function-name ${FUNCTION_NAME} \\"
echo "    --payload '{\"task\":\"ingest_load_fuel\"}' \\"
echo "    --cli-binary-format raw-in-base64-out /tmp/out.json \\"
echo "    --region ${REGION} && cat /tmp/out.json"
echo "==================================================================="
