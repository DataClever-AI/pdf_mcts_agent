#!/usr/bin/env bash
# One-time setup: private ECR repo + read-only IAM policy for QA pulls.
# Idempotent — safe to re-run, existing resources are skipped with a warning.
#
# Required env vars: AWS_REGION, AWS_ACCOUNT_ID
# Usage:
#   AWS_REGION=us-east-1 AWS_ACCOUNT_ID=123456789012 bash scripts/aws_ecr_setup.sh
set -euo pipefail

: "${AWS_REGION:?set AWS_REGION, e.g. us-east-1}"
: "${AWS_ACCOUNT_ID:?set AWS_ACCOUNT_ID, e.g. 123456789012}"

REPO_NAME=mvp-v2
POLICY_NAME=mvp-v2-qa-ecr-pull
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_DIR="$SCRIPT_DIR/../aws"

echo "==> Creating ECR repository ($REPO_NAME) in $AWS_REGION"
aws ecr create-repository \
  --repository-name "$REPO_NAME" \
  --region "$AWS_REGION" \
  --image-scanning-configuration scanOnPush=true \
  --image-tag-mutability IMMUTABLE \
  || echo "    (repository already exists, continuing)"

echo "==> Applying lifecycle policy (expire untagged after 7d, keep last 10 tagged)"
aws ecr put-lifecycle-policy \
  --repository-name "$REPO_NAME" \
  --region "$AWS_REGION" \
  --lifecycle-policy-text "file://$AWS_DIR/lifecycle-policy.json"

echo "==> Creating QA pull-only IAM policy ($POLICY_NAME)"
sed -e "s/AWS_REGION/$AWS_REGION/" -e "s/AWS_ACCOUNT_ID/$AWS_ACCOUNT_ID/" \
  "$AWS_DIR/qa-ecr-pull-policy.json" > /tmp/qa-ecr-pull-policy.rendered.json

aws iam create-policy \
  --policy-name "$POLICY_NAME" \
  --policy-document "file:///tmp/qa-ecr-pull-policy.rendered.json" \
  || echo "    (policy already exists, continuing)"

cat <<EOF

==> Setup done. To create/onboard a QA member:

  QA_USER=qa-<name>
  aws iam create-user --user-name "\$QA_USER"
  aws iam attach-user-policy --user-name "\$QA_USER" \\
      --policy-arn arn:aws:iam::$AWS_ACCOUNT_ID:policy/$POLICY_NAME
  aws iam create-access-key --user-name "\$QA_USER"

The create-access-key output (AccessKeyId + SecretAccessKey) is shown ONCE —
share it with that QA member over a secure channel, never in chat/email/git.

QA then runs, once:
  aws configure --profile mvp-v2-qa   # paste the access key + region: $AWS_REGION
  aws ecr get-login-password --region $AWS_REGION --profile mvp-v2-qa | \\
      docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

And to run the app (repeat whenever the ECR token expires, ~12h):
  docker pull $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:<tag>
  docker run --rm -p 8501:8501 --env-file .env $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:<tag>
EOF
