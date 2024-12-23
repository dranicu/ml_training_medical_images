import json
import requests
import os

from io import BytesIO


def handler(ctx, data: BytesIO = None):
    cfg = ctx.Config()
    try:
        ARGO_SERVER_URL = os.getenv('ARGO_SERVER_URL')  # Fallback if variable is not set
        ARGO_TOKEN = os.getenv('ARGO_TOKEN')
    except Exception as e:
        print(f"Error: {e}")
        
    try:
        headers={
                "Authorization": f"Bearer {ARGO_TOKEN}",
                "Content-Type": "application/json"
            }
        json_data = {
            'namespace': 'argo-workflows',
            'resourceKind': 'WorkflowTemplate',
            'resourceName': 'webhook-triggered-ml',
            }
        # Trigger Argo workflow using a POST request to Argo's workflow submit endpoint
        response = requests.post(
            ARGO_SERVER_URL,
            headers=headers,
            json=json_data,
            verify=False
        )

        if response.status_code == 200:
            print("Successfully triggered Argo workflow.")
            return {"status": "success", "data": response.json()}
        else:
            print(f"Failed to trigger Argo workflow: {response.text}")
            return {"status": "failure", "data": response.text}
        
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "failure", "error": str(e)}