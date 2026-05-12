#!/usr/bin/env bash
# IntelFactor — Datadog AWS Essentials Setup
# Deploys the Datadog AWS Integration IAM role + Datadog Forwarder Lambda.
#
# Usage:
#   export DD_API_KEY="your-datadog-api-key"
#   export DD_EXTERNAL_ID="your-external-id-from-datadog"
#   ./setup-datadog-aws.sh
#
# Prerequisites:
#   - AWS CLI configured with sufficient IAM/CloudFormation permissions
#   - External ID generated from Datadog AWS integration page
#
# Resources created:
#   - IAM role for Datadog to pull AWS metrics/events
#   - Lambda function to forward CloudWatch logs to Datadog
#   - S3 bucket for Forwarder code storage
#   - Secrets Manager secret for API key

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-west-2}"
STACK_PREFIX="${STACK_PREFIX:-intelfactor}"
DD_API_KEY="${DD_API_KEY:-}"
DD_APP_KEY="${DD_APP_KEY:-}"
DD_SITE="${DD_SITE:-datadoghq.com}"
DD_EXTERNAL_ID="${DD_EXTERNAL_ID:-}"

if [[ -z "$DD_API_KEY" ]]; then
    echo "ERROR: DD_API_KEY is required. Set it as an environment variable."
    exit 1
fi

if [[ -z "$DD_EXTERNAL_ID" ]]; then
    echo "ERROR: DD_EXTERNAL_ID is required."
    echo "Generate one in Datadog: Integrations -> AWS -> Add AWS Account -> Role Delegation"
    exit 1
fi

echo "=========================================="
echo "IntelFactor Datadog AWS Essentials Setup"
echo "Region: $AWS_REGION"
echo "Site: $DD_SITE"
echo "=========================================="

# ── 1. Deploy AWS Integration ──────────────────────────────────────────
echo ""
echo "[1/2] Deploying Datadog AWS Integration..."

INTEGRATION_STACK="${STACK_PREFIX}-datadog-integration"

aws cloudformation deploy \
    --stack-name "$INTEGRATION_STACK" \
    --template-file cloudformation-datadog-integration.yaml \
    --region "$AWS_REGION" \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameter-overrides \
        ExternalId="$DD_EXTERNAL_ID" \
        DdApiKey="$DD_API_KEY" \
        DdAppKey="$DD_APP_KEY" \
        DdSite="$DD_SITE" \
    --no-fail-on-empty-changeset

INTEGRATION_ROLE_ARN=$(aws cloudformation describe-stacks \
    --stack-name "$INTEGRATION_STACK" \
    --region "$AWS_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='IntegrationRoleArn'].OutputValue" \
    --output text)

echo "✓ Integration Role ARN: $INTEGRATION_ROLE_ARN"
echo "  → Add this ARN in Datadog: Integrations -> AWS -> Add AWS Account -> Role Delegation"

# ── 2. Deploy Datadog Forwarder ────────────────────────────────────────
echo ""
echo "[2/2] Deploying Datadog Forwarder..."

FORWARDER_STACK="${STACK_PREFIX}-datadog-forwarder"

aws cloudformation deploy \
    --stack-name "$FORWARDER_STACK" \
    --template-file cloudformation-datadog-forwarder.yaml \
    --region "$AWS_REGION" \
    --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
    --parameter-overrides \
        DdApiKey="$DD_API_KEY" \
        DdSite="$DD_SITE" \
        FunctionName="${STACK_PREFIX}-datadog-forwarder" \
    --no-fail-on-empty-changeset

FORWARDER_ARN=$(aws cloudformation describe-stacks \
    --stack-name "$FORWARDER_STACK" \
    --region "$AWS_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='ForwarderArn'].OutputValue" \
    --output text)

echo "✓ Forwarder Lambda ARN: $FORWARDER_ARN"
echo "  → Subscribe this to CloudWatch log groups or S3 buckets to forward logs"

# ── Summary ────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Setup Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. In Datadog, go to Integrations -> AWS -> Add AWS Account"
echo "  2. Choose 'Role Delegation' and enter:"
echo "     - Account ID: $(aws sts get-caller-identity --query Account --output text)"
echo "     - Role Name: IntelFactorDatadogIntegrationRole"
echo "     - External ID: $DD_EXTERNAL_ID"
echo ""
echo "  3. Subscribe the Forwarder to log sources:"
echo "     aws logs put-subscription-filter \\"
echo "       --log-group-name '/aws/ecs/containerinsights/<cluster>/performance' \\"
echo "       --filter-name datadog-forwarder \\"
echo "       --filter-pattern '' \\"
echo "       --destination-arn $FORWARDER_ARN"
echo ""
echo "  4. For S3 logs:"
echo "     aws s3api put-bucket-notification-configuration \\"
echo "       --bucket <bucket-name> \\"
echo "       --notification-configuration '{\"LambdaFunctionConfigurations\":[{\"LambdaFunctionArn\":\"$FORWARDER_ARN\",\"Events\":[\"s3:ObjectCreated:*\"]}]}'"
echo ""
echo "Stacks:"
echo "  - $INTEGRATION_STACK"
echo "  - $FORWARDER_STACK"
