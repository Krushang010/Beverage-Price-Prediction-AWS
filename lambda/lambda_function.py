import json
import os
import secrets

import boto3


ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
API_SHARED_SECRET = os.environ["API_SHARED_SECRET"]

runtime = boto3.client("sagemaker-runtime")


def build_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(body),
    }


def get_header(headers, header_name):
    if not headers:
        return None

    for key, value in headers.items():
        if key.lower() == header_name.lower():
            return value

    return None


def lambda_handler(event, context):

    try:
        # --------------------------------------------------
        # Authentication
        # --------------------------------------------------
        headers = event.get("headers") or {}

        provided_secret = (
            get_header(headers, "x-api-key")
            or ""
        )

        if not secrets.compare_digest(
            str(provided_secret),
            API_SHARED_SECRET,
        ):
            return build_response(
                401,
                {"error": "Unauthorized"},
            )

        # --------------------------------------------------
        # Parse API Gateway request body
        # --------------------------------------------------
        body = event.get("body")

        if isinstance(body, str):
            payload = json.loads(body)

        elif isinstance(body, dict):
            payload = body

        else:
            return build_response(
                400,
                {"error": "Invalid request body"},
            )

        # --------------------------------------------------
        # Invoke SageMaker endpoint
        # --------------------------------------------------
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload).encode("utf-8"),
        )

        result = json.loads(
            response["Body"]
            .read()
            .decode("utf-8")
        )

        return build_response(
            200,
            result,
        )

    except json.JSONDecodeError:
        return build_response(
            400,
            {"error": "Invalid JSON"},
        )

    except Exception as error:

        # Detailed error goes only to CloudWatch
        print(
            f"Prediction error: "
            f"{type(error).__name__}: {error}"
        )

        # Do not expose internal details publicly
        return build_response(
            500,
            {"error": "Internal server error"},
        )