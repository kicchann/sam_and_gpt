import json
from typing import List, Optional

from openai import AzureOpenAI

from .gpt_config import (
    AZURE_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
)

FUNCTION_NAME = "attach_labels_to_image"


class GPTLabelCreator:
    def __init__(self, client: Optional[AzureOpenAI] = None):
        if client is None:
            self._client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_KEY,
                api_version=AZURE_API_VERSION,
            )
        else:
            self._client = client

    def create_label_w_annotated_image(
        self,
        data_url: str,
        label_suggestions: List[str],
        suggestions_w_remark: List[str],
    ):
        assert len(label_suggestions) > 0, "label_suggestions must not be empty"
        # suggestions_w_remarkにはlabal_suggestions以外のラベルを指定してはいけない
        assert all(
            [label in label_suggestions for label in suggestions_w_remark]
        ), "suggestions_w_remark should be subset of label_suggestions"

        system_message = f"""
        You are required to review the input image to label the objects that are highlighted with red indexes and boundaries considering their surrounding context.
        Choose your labels from the following list: {', '.join([f"'{label}'" for label in label_suggestions])}
        If your chosen label is included in this list: {', '.join([f"'{label}'" for label in suggestions_w_remark])},
        please provide additional details in the 'remark' field. This could include any other relevant information about the object.
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": FUNCTION_NAME,
                    "description": "create labels with an image according to numbers and segmentations on it",
                    "parameters": {
                        "type": "object",
                        # labelのリストを出力
                        "properties": {
                            "labels": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index": {
                                            "type": "integer",
                                        },
                                        "label": {
                                            "type": "string",
                                        },
                                        "remark": {
                                            "type": "string",
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            }
        ]

        completion = self._client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    ],
                },
            ],
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": FUNCTION_NAME},
            },  # auto is default, but we'll be explicit
        )

        response_message = completion.choices[0].message
        tool_calls = response_message.tool_calls

        results = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            if function_name == FUNCTION_NAME:
                results.extend(function_args["labels"])
        return results
