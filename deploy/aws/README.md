# IntelFactor — Datadog on AWS

This directory contains AWS CloudFormation templates and scripts for deploying the complete Datadog observability stack for IntelFactor.

## What's Included

| Component | File | Purpose |
|-----------|------|---------|
| **AWS Integration** | `cloudformation-datadog-integration.yaml` | IAM role allowing Datadog to pull CloudWatch metrics/events from your AWS account |
| **Datadog Forwarder** | `cloudformation-datadog-forwarder.yaml` | Lambda function that forwards CloudWatch Logs, S3 events, and SNS messages to Datadog |
| **ECS Agent** | `cloudformation-ecs-datadog.yaml` | ECS daemon service running the Datadog Agent with ECS Explorer enabled |
| **ECS Task Definitions** | `ecs-datadog-*.json` | Standalone ECS task definitions for EC2 launch type |
| **Setup Script** | `setup-datadog-aws.sh` | One-command deployment of Integration + Forwarder |

---

## Prerequisites

- AWS CLI configured with sufficient permissions
- Datadog account with API key
- External ID from Datadog AWS integration page (for the Integration stack)

---

## Quick Start: Deploy All Essentials

```bash
cd deploy/aws

export DD_API_KEY="your-datadog-api-key"
export DD_EXTERNAL_ID="your-external-id-from-datadog"
export DD_SITE="datadoghq.com"   # or datadoghq.eu, us3.datadoghq.com, etc.
export AWS_REGION="us-west-2"

chmod +x setup-datadog-aws.sh
./setup-datadog-aws.sh
```

This deploys:
1. **AWS Integration IAM Role** — Datadog can now collect EC2, ECS, ELB, RDS, S3, Lambda, etc. metrics
2. **Datadog Forwarder Lambda** — ready to subscribe to CloudWatch log groups or S3 buckets

After the script finishes, copy the Integration Role ARN into Datadog:
**Integrations → AWS → Add AWS Account → Role Delegation**

---

## Component-by-Component Deployment

### 1. AWS Integration (IAM Role)

Creates the IAM role that Datadog assumes to read your AWS resources.

```bash
aws cloudformation deploy \
  --stack-name intelfactor-datadog-integration \
  --template-file cloudformation-datadog-integration.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ExternalId="your-external-id" \
    DdApiKey="your-api-key" \
    DdSite="datadoghq.com"
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ExternalId` | (required) | External ID from Datadog AWS integration page |
| `DdApiKey` | (required) | Your Datadog API key |
| `DdAppKey` | (empty) | Datadog APP key (optional) |
| `DdSite` | `datadoghq.com` | Datadog site |
| `EnableCloudSecurity` | `false` | Enable Cloud Security Misconfigurations (CSPM) |
| `EnableCloudCost` | `false` | Enable Cloud Cost Management |

**Outputs:**
- `IntegrationRoleArn` — Paste this into Datadog's AWS integration page
- `IntegrationRoleName` — `IntelFactorDatadogIntegrationRole`

---

### 2. Datadog Forwarder (Lambda)

Forwards logs from CloudWatch, S3, and SNS to Datadog.

```bash
aws cloudformation deploy \
  --stack-name intelfactor-datadog-forwarder \
  --template-file cloudformation-datadog-forwarder.yaml \
  --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
  --parameter-overrides \
    DdApiKey="your-api-key" \
    DdSite="datadoghq.com" \
    FunctionName="intelfactor-datadog-forwarder"
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DdApiKey` | (required) | Datadog API key |
| `DdApiKeySecretArn` | (empty) | Alternative: ARN of Secrets Manager secret |
| `DdSite` | `datadoghq.com` | Datadog site |
| `DdTags` | `env:production,project:intelfactor` | Tags applied to all forwarded data |
| `FunctionName` | `intelfactor-datadog-forwarder` | Lambda function name |
| `MemorySize` | `1024` | Lambda memory (MB) |
| `Timeout` | `120` | Lambda timeout (seconds) |

**Outputs:**
- `ForwarderArn` — Subscribe this to log sources
- `ApiKeySecretArn` — Secret storing the API key

**Subscribing the Forwarder:**

```bash
# CloudWatch Logs (e.g. ECS container insights)
aws logs put-subscription-filter \
  --log-group-name "/aws/ecs/containerinsights/intelfactor-cluster/performance" \
  --filter-name datadog-forwarder \
  --filter-pattern "" \
  --destination-arn "arn:aws:lambda:<region>:<account>:function:intelfactor-datadog-forwarder"

# S3 bucket logs
aws s3api put-bucket-notification-configuration \
  --bucket my-bucket \
  --notification-configuration '{
    "LambdaFunctionConfigurations": [{
      "LambdaFunctionArn": "arn:aws:lambda:<region>:<account>:function:intelfactor-datadog-forwarder",
      "Events": ["s3:ObjectCreated:*"]
    }]
  }'
```

> **Note:** For production, use the official Datadog Forwarder from the AWS Serverless Application Repository or Datadog's published CloudFormation template at `https://datadog-cloudformation-template.s3.amazonaws.com/aws/forwarder/latest.yaml`. The template here is a lightweight project-specific wrapper.

---

### 3. ECS Agent with ECS Explorer

Deploys the Datadog Agent as an ECS daemon service for container-level monitoring.

```bash
aws cloudformation create-stack \
  --stack-name intelfactor-datadog-ecs \
  --template-body file://cloudformation-ecs-datadog.yaml \
  --parameters \
    ParameterKey=ClusterName,ParameterValue=intelfactor-cluster \
    ParameterKey=DatadogApiKey,ParameterValue=$DD_API_KEY \
    ParameterKey=DatadogSite,ParameterValue=$DD_SITE \
  --capabilities CAPABILITY_NAMED_IAM
```

Or use the task definitions directly:

```bash
# Minimal (ECS Explorer + metrics)
aws ecs register-task-definition \
  --cli-input-json file://ecs-datadog-task-definition.json

# Full (adds APM, Logs, DogStatsD, Process monitoring)
aws ecs register-task-definition \
  --cli-input-json file://ecs-datadog-full-task-definition.json

# Deploy as daemon
aws ecs create-service \
  --cluster intelfactor-cluster \
  --service-name datadog-agent \
  --task-definition intelfactor-datadog-agent-full \
  --scheduling-strategy DAEMON
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AWS Account                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────┐    ┌───────────────────────────┐   │
│  │  Datadog AWS        │    │  Datadog Forwarder        │   │
│  │  Integration Role   │◄───│  (Lambda)                 │   │
│  │                     │    │                           │   │
│  │  • CloudWatch Read  │    │  • CloudWatch Logs        │   │
│  │  • EC2 Describe     │    │  • S3 Events              │   │
│  │  • ECS Describe     │    │  • SNS Events             │   │
│  │  • S3 Read          │    │                           │   │
│  │  • Lambda List      │    │  ───────► Datadog         │   │
│  └─────────────────────┘    └───────────────────────────┘   │
│           │                              ▲                   │
│           │                              │                   │
│           ▼                              │                   │
│  ┌───────────────────────────────────────┴──────────┐       │
│  │              IntelFactor ECS Cluster              │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │       │
│  │  │ Station  │  │  Sync    │  │ Datadog Agent│───┘       │
│  │  │ Service  │  │  Agent   │  │ (Daemon)     │           │
│  │  └──────────┘  └──────────┘  └──────────────┘           │
│  └────────────────────────────────────────────────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Datadog       │
                    │   (metrics +    │
                    │    logs + APM)  │
                    └─────────────────┘
```

---

## Security Notes

- **External ID**: Always use an external ID for the AWS integration role to prevent the [confused deputy problem](https://docs.aws.amazon.com/IAM/latest/UserGuide/confused-deputy.html).
- **API Keys**: The Forwarder templates support plaintext keys or AWS Secrets Manager ARNs. Use Secrets Manager in production.
- **KMS**: The Forwarder template includes `kms:Decrypt` for all resources to handle KMS-encrypted S3 buckets. You can safely remove this after installation if not needed.
- **Ports**: DogStatsD (`8125/udp`) and APM (`8126/tcp`) on ECS are bound to the host interface. Ensure EC2 security groups block these from the public internet.

---

## Cost Considerations

- **Datadog Forwarder Lambda**: Charged per invocation and GB-second. Log volume drives cost.
- **CloudWatch Logs ingestion**: AWS charges for data ingestion; the Forwarder adds a subscription filter but does not change ingestion pricing.
- **Datadog AWS Integration**: No additional AWS cost; Datadog may charge per host or custom metric depending on your plan.

---

## Further Reading

- [Datadog AWS Integration Docs](https://docs.datadoghq.com/getting_started/integrations/aws/)
- [Datadog Forwarder Docs](https://docs.datadoghq.com/logs/guide/forwarder/)
- [Amazon Elastic Container Explorer](https://docs.datadoghq.com/containers/monitoring/amazon_elastic_container_explorer/)
- [ECS Log Collection](https://docs.datadoghq.com/containers/amazon_ecs/logs/)
- [AWS Organizations Multi-Account Setup](https://docs.datadoghq.com/integrations/guide/aws-organizations-setup/)
