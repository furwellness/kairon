import ujson as json
import logging
import typing
from typing import Any, Dict, List, Optional, Text
from abc import ABC
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
import faiss
import rasa.utils.io as io_utils
import os
from rasa.shared.nlu.constants import TEXT, INTENT
import litellm
import numpy as np
from tqdm import tqdm
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    pass


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class LLMClassifier(GraphComponent, IntentClassifier, ABC):
    """Intent and Entity classifier using the OpenAI Completion framework"""

    system_prompt = "You will be provided with a text, and your task is to classify its intent as {0}. Provide output in json format with the following keys intent, explanation, text."

    def __init__(
        self,
        config: Optional[Dict[Text, Any]],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        vector: Optional[faiss.IndexFlatIP] = None,
        data: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Construct a new intent classifier using the OpenAI Completion framework."""
        self.api_key = None
        self.component_config = config
        self._model_storage = model_storage
        self._resource = resource
        self._execution_context = execution_context
        self.load_api_key(config.get("bot_id"))
        self.vector = vector
        self.data = data
    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["litellm", "numpy"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            "bot_id": None,
            "prediction_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "embedding_size": 1536,
            "top_k": 5,
            "temperature": 0.0,
            "max_tokens": 50,
            "retry": 3,
        }

    def load_api_key(self, bot_id: Text):
        if bot_id:
            from kairon.shared.admin.processor import Sysadmin
            llm_secret = Sysadmin.get_llm_secret("openai", bot_id)
            self.api_key = llm_secret.get('api_key')
        elif os.environ.get("API_KEY"):
            self.api_key = os.environ.get("API_KEY")
        else:
            raise KeyError(
                f"either set bot_id'in OpenAIClassifier config or set OPENAI_API_KEY in environment variables"
            )

    def get_embeddings(self, text):
        embedding = litellm.embedding(
            model=self.component_config.get("embedding_model", "text-embedding-3-small"), input=text, api_key=self.api_key
        )["data"][0]["embedding"]
        return embedding

    def train(self, training_data: TrainingData) -> Resource:
        """Train the intent classifier on a data set."""
        data_map = []
        vector_map = []
        for example in tqdm(training_data.intent_examples):
            vector_map.append(self.get_embeddings(example.get(TEXT)))
            data_map.append({"text": example.get(TEXT), "intent": example.get(INTENT)})
        np_vector = np.asarray(vector_map, dtype=np.float32)
        self.vector = faiss.IndexFlatIP(len(vector_map[0]))
        self.vector.add(np_vector)
        self.data = data_map
        self.persist()
        return self._resource

    def prepare_context(self, embeddings, text):
        dist, indx = self.vector.search(
            np.asarray([embeddings], dtype=np.float32),
            k=self.component_config.get("top_k", 5),
        )
        context = ""
        intents = {"nlu_fallback"}
        for idx, value in enumerate(indx[0]):
            if dist[0][idx] >= 0.7:
                context += f"text: {self.data[value]['text']}\nintent: {self.data[value]['intent']}\n\n"
                intents.add(self.data[value]['intent'])
        messages = [
            {"role": "system", "content": self.system_prompt.format(intents)},
            {
                "role": "user",
                "content": f'''##{self.system_prompt.format(intents)}
                ##Based on the Intent Context generate the intent.
                If intent must belongs to {intents}
                Intent Context:
                {context}
                text: {text}
                intent: 
                ''',
            }
        ]
        return messages, intents

    def predict(self, text):
        embedding = self.get_embeddings(text)
        messages, intents = self.prepare_context(embedding, text)
        response = litellm.completion(
            model=self.component_config.get("prediction_model", "gpt-4o-mini"),
            messages=messages,
            temperature=self.component_config.get("temperature", 0.0),
            max_tokens=self.component_config.get("max_tokens", 50),
            api_key=self.api_key,
            max_retries=self.component_config.get("retry", 3)
        )
        logger.debug(response)
        print(response)
        responses = json.loads(response.choices[0]["message"]["content"])
        intent = responses["intent"] if "intent" in responses.keys() else None
        if intent not in intents:
            explanation = f"invalid intent predicted {intent}, falling back to nlu_fallback"
            intent = "nlu_fallback"
        else:
            explanation = (
                responses["explanation"]
                if "explanation" in responses.keys()
                else None
            )
        return intent, explanation

    def process(self, messages: List[Message]) -> List[Message]:
        """Return the most likely intent and its probability for a message."""
        for message in messages:
            if not self.vector and not self.data:
                # component is either not trained or didn't
                # receive enough training data
                intent = None
                intent_ranking = []
            else:
                label, reason = self.predict(message.get(TEXT))
                intent = {"name": label, "confidence": 1, "reason": reason}
                intent_ranking = []
            message.set("intent", intent, add_to_output=True)
            message.set("intent_ranking", intent_ranking, add_to_output=True)
        return messages

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "LLMClassifier":
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource, execution_context)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> "LLMClassifier":
        """Loads a policy from the storage (see parent class for full docstring)."""
        try:
            with model_storage.read_from(resource) as model_path:
                file_name = cls.__name__
                vector_file_name = file_name + "_vector.db"
                data_file_name = file_name + "_data.pkl"
                vector_file = os.path.join(model_path, vector_file_name)
                data_file = os.path.join(model_path, data_file_name)

                vector = faiss.read_index(vector_file)
                data = io_utils.json_unpickle(data_file)
                return cls(
                    config, model_storage, resource, execution_context, vector, data
                )
        except ValueError:
            logger.debug(
                f"Failed to load {cls.__class__.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
        return cls(config, model_storage, resource, execution_context)

    def persist(self) -> None:
        """Persist this model into the passed directory."""
        with self._model_storage.write_to(self._resource) as model_path:
            file_name = self.__class__.__name__
            vector_file_name = file_name + "_vector.db"
            data_file_name = file_name + "_data.pkl"
            if self.vector and self.data:
                faiss.write_index(
                    self.vector, os.path.join(model_path, vector_file_name)
                )
                io_utils.json_pickle(
                    os.path.join(model_path, data_file_name), self.data
                )
