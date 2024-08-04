import semanticKernel
import azureopenai


import logging
import asyncio
from decouple import config
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from typing import Annotated
from semantic_kernel.functions import kernel_function

# Initialize the kernel
kernel = Kernel()

# Add Azure OpenAI chat completion
kernel.add_service(
    AzureChatCompletion(
        deployment_name="your_models_deployment_name",
        api_key="your_api_key",
        base_url="your_base_url",
    )
)

# Set the logging level for semantic_kernel.kernel to DEBUG
logging.basicConfig(
    format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("semantic_kernel.kernel").setLevel(logging.DEBUG)

# Retrieve the chat completion service
chat_completion: AzureChatCompletion = kernel.get_service(type=ChatCompletionClientBase)

class LightsPlugin:
    lights = [
        {"id": 1, "name": "Table Lamp", "is_on": False},
        {"id": 2, "name": "Porch light", "is_on": False},
        {"id": 3, "name": "Chandelier", "is_on": True},
    ]

    @kernel_function(
        name="get_lights",
        description="Gets a list of lights and their current state",
    )
    def get_state(self) -> Annotated[str, "the output is a string"]:
        """Gets a list of lights and their current state."""
        return str(self.lights)

    @kernel_function(
        name="change_state",
        description="Changes the state of the light",
    )
    def change_state(self, id: int, is_on: bool) -> Annotated[str, "the output is a string"]:
        """Changes the state of the light."""
        for light in self.lights:
            if light["id"] == id:
                light["is_on"] = is_on
                return str(light)
        return "Light not found"

# Add the plugin to the kernel
kernel.add_plugin(
    LightsPlugin(),
    plugin_name="Lights",
)

execution_settings = AzureChatPromptExecutionSettings(tool_choice="auto")
execution_settings.function_call_behavior = FunctionCallBehavior.EnableFunctions(auto_invoke=True, filters={})

# Create a history of the conversation
history = ChatHistory()

async def get_ai_response():
    result = await chat_completion.get_chat_message_contents(
        chat_history=history,
        settings=execution_settings,
        kernel=kernel,
        arguments=KernelArguments(),
    )
    return result[0]

# Run the async function to get the response
response = asyncio.run(get_ai_response())
print(response)

#API 
#Small Key
