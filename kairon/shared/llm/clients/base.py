from abc import ABC
from typing import Text


class LLMResources(ABC):

    async def invoke(self, resource: Text, engine: Text, **kwargs):
        raise NotImplementedError("Provider not implemented")
