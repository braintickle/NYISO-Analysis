#!/usr/bin/env bash
# setup_ecs.sh — One-time ECS Fargate infrastructure setup for NYISO dashboard
#
# What this creates:
#   - ECR repository:           nyiso-streamlit
#   - CloudWatch log group:     /ecs/nyiso-dashboard
#   - IAM task execution role:  nyiso-ecs-exec-role
#   - Security group:           nyiso-dashboard-sg  (inbound :8501, all outbound)
#   - ECS cluster:              nyiso-dashboard
#   - ECS task definition:      nyiso-dashboard  (Fargate, 0.5 vCPU / 1 GB)
#   - ECS service:              nyiso-dashboard  (desired=1, public IP)
#
# Run once from repo root after attaching iam/nyiso-deploy-policy.json to nyiso-deploy:
#   bash scripts/setup_ecs.sh
#
# After this runs, push to GitHub to trigger the first real deployment via
# .github/workflows/deploy.yml (which replaces the placeholder nginx image).

set -euo pipefail

REGION="us-east-1"
ACCOUNT="358592704128"
VPC_ID="vpc-06b70275d8de36415"
# Two public subnets for HA
SUBNET_1="subnet-060f12554b8f1ea8c"   # us-east-1a
SUBNET_2="subnet-0397783105cb80f38"   # us-east-1b

ECR_REPO="nyiso-streamlit"
CLUSTER="nyiso-dashboard"
SERVICE="nyiso-dashboard"
TASK_FAMILY="nyiso-dashboard"
SG_NAME="nyiso-dashboard-sg"
EXEC_ROLE="nyiso-ecs-exec-role"
LOG_GROUP="/ecs/nyiso-dashboard"
ECR_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"

set -a; source .env; set +a
if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "ERROR: DATABASE_URL not set in .env" >&2; exit 1
fi

echo "=== Step 1: ECR repository ==="
aws ecr create-repository \
  --repository-name "${ECR_REPO}" \
  --region "${REGION}" \
  --image-scanning-configuration scanOnPush=true \
  2>/dev/null && echo "  Created ${ECR_REPO}" || echo "  Already exists — skipping."

echo ""
echo "=== Step 2: CloudWatch log group ==="
aws logs create-log-group \
  --log-group-name "${LOG_GROUP}" \
  --region "${REGION}" \
  2>/dev/null && echo "  Created ${LOG_GROUP}" || echo "  Already exists — skipping."

echo ""
echo "=== Step 3: ECS task execution IAM role ==="
aws iam get-role --role-name "${EXEC_ROLE}" > /dev/null 2>&1 && \
  echo "  Role already exists — skipping." || \
  aws iam create-role \
    --role-name "${EXEC_ROLE}" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{
        "Effect":"Allow",
        "Principal":{"Service":"ecs-tasks.amazonaws.com"},
        "Action":"sts:AssumeRole"
      }]
    }' \
    --description "ECS task execution role for NYISO dashboard"

aws iam attach-role-policy \
  --role-name "${EXEC_ROLE}" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
  2>/dev/null || true

EXEC_ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${EXEC_ROLE}"
echo "  Exec role ARN: ${EXEC_ROLE_ARN}"

echo ""
echo "=== Step 4: Security group ==="
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=${SG_NAME}" "Name=vpc-id,Values=${VPC_ID}" \
  --query 'SecurityGroups[0].GroupId' --output text --region "${REGION}" 2>/dev/null)

if [[ "${SG_ID}" == "None" || -z "${SG_ID}" ]]; then
  SG_ID=$(aws ec2 create-security-group \
    --group-name "${SG_NAME}" \
    --description "NYISO Streamlit dashboard - allow port 8501 inbound" \
    --vpc-id "${VPC_ID}" \
    --region "${REGION}" \
    --query 'GroupId' --output text)
  echo "  Created security group: ${SG_ID}"

  # Allow Streamlit port from anywhere
  aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" \
    --protocol tcp --port 8501 \
    --cidr 0.0.0.0/0 \
    --region "${REGION}"

  # Allow all outbound (Supabase + NYISO API + Open-Meteo)
  aws ec2 authorize-security-group-egress \
    --group-id "${SG_ID}" \
    --protocol -1 --port -1 \
    --cidr 0.0.0.0/0 \
    --region "${REGION}" \
    2>/dev/null || true
else
  echo "  Security group already exists: ${SG_ID}"
fi

echo ""
echo "=== Step 5: ECS cluster ==="
aws ecs create-cluster \
  --cluster-name "${CLUSTER}" \
  --region "${REGION}" \
  --capacity-providers FARGATE \
  --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1 \
  2>/dev/null && echo "  Created cluster: ${CLUSTER}" || echo "  Already exists."

echo ""
echo "=== Step 6: Task definition (placeholder nginx image — GitHub Actions will update) ==="
TASK_DEF=$(cat <<JSON
{
  "family": "${TASK_FAMILY}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "${EXEC_ROLE_ARN}",
  "containerDefinitions": [
    {
      "name": "nyiso-streamlit",
      "image": "nginx:alpine",
      "portMappings": [
        {"containerPort": 8501, "protocol": "tcp"}
      ],
      "environment": [
        {"name": "DATABASE_URL", "value": "${DATABASE_URL}"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "${LOG_GROUP}",
          "awslogs-region": "${REGION}",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "essential": true
    }
  ]
}
JSON
)

TASK_DEF_ARN=$(aws ecs register-task-definition \
  --cli-input-json "${TASK_DEF}" \
  --region "${REGION}" \
  --query 'taskDefinition.taskDefinitionArn' --output text)
echo "  Task definition: ${TASK_DEF_ARN}"

echo ""
echo "=== Step 7: ECS service ==="
SVC_EXISTS=$(aws ecs describe-services \
  --cluster "${CLUSTER}" --services "${SERVICE}" \
  --region "${REGION}" \
  --query 'services[?status==`ACTIVE`].serviceName' --output text 2>/dev/null)

if [[ -n "${SVC_EXISTS}" ]]; then
  echo "  Service already exists — updating..."
  aws ecs update-service \
    --cluster "${CLUSTER}" \
    --service "${SERVICE}" \
    --task-definition "${TASK_DEF_ARN}" \
    --region "${REGION}" > /dev/null
else
  aws ecs create-service \
    --cluster "${CLUSTER}" \
    --service-name "${SERVICE}" \
    --task-definition "${TASK_DEF_ARN}" \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={
      subnets=[${SUBNET_1},${SUBNET_2}],
      securityGroups=[${SG_ID}],
      assignPublicIp=ENABLED
    }" \
    --region "${REGION}" > /dev/null
  echo "  Service created: ${SERVICE}"
fi

echo ""
echo "==================================================================="
echo "ECS infrastructure ready!"
echo ""
echo "  Cluster:    ${CLUSTER}"
echo "  Service:    ${SERVICE}"
echo "  ECR repo:   ${ECR_URI}"
echo "  Security:   ${SG_ID}  (port 8501 open)"
echo ""
echo "Next: add GitHub Secrets, then push to main."
echo "  GitHub secrets required:"
echo "    AWS_ACCESS_KEY_ID      — nyiso-deploy access key"
echo "    AWS_SECRET_ACCESS_KEY  — nyiso-deploy secret key"
echo "    DATABASE_URL           — ${DATABASE_URL:0:40}..."
echo ""
echo "After first GitHub Actions run, get the public IP:"
echo "  TASK_ARN=\$(aws ecs list-tasks --cluster ${CLUSTER} --query 'taskArns[0]' --output text --region ${REGION})"
echo "  aws ecs describe-tasks --cluster ${CLUSTER} --tasks \$TASK_ARN --region ${REGION} --query 'tasks[0].attachments[0].details'"
echo "==================================================================="
