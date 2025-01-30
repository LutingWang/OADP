import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, BatchEncoding
from transformers import MllamaProcessor, AutoTokenizer, PreTrainedTokenizerFast

class Chatbot:
    PRETRAINED = 'pretrained/llama/Llama-3.2-11B-Vision-Instruct'

    def __init__(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.PRETRAINED)
        self._tokenizer: PreTrainedTokenizerFast = tokenizer

        processor = MllamaProcessor.from_pretrained(self.PRETRAINED)
        self._processor = processor

        model = MllamaForConditionalGeneration.from_pretrained(
            self.PRETRAINED,
            device_map='auto',
            torch_dtype='auto',
        )
        self._model = model

    def __call__(self, inputs: BatchEncoding) -> str:
        inputs = inputs.to('cuda')

        input_ids: torch.Tensor = inputs['input_ids']
        _, input_length = input_ids.shape

        output_ids = self._model.generate(**inputs, max_new_tokens=1024)
        generated_ids = output_ids[0, input_length:]
        generated_text = self._processor.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return generated_text

    def chat(self, text: str) -> str:
        conversation = [dict(role='user', content=text)]
        inputs = self._tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors='pt',
            return_dict=True,
        )
        return self(inputs)

    def chat_multimodal(self, images: list[Image.Image], text: str) -> str:
        content = [dict(type='image') for _ in images]
        content.append(dict(type='text', text=text))
        conversation = [dict(role='user', content=content)]
        input_text = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            images,
            input_text,
            add_special_tokens=False,
            return_tensors='pt',
        )
        return self(inputs)

class MutiCaptioner:
    PROMPT = "Answer with a template of the form: A photo of <object>, example: A photo of cat."
    def __init__(self, chatbot: Chatbot) -> None:
        self._chatbot = chatbot

    def __call__(self, images: list[Image.Image]) -> str:
        return self._chatbot.chat_multimodal(images, self.PROMPT)


class Captioner:
    PROMPT = "Describe the primary objects in a single sentence."

    def __init__(self, chatbot: Chatbot) -> None:
        self._chatbot = chatbot

    def __call__(self, image: Image.Image) -> str:
        return self._chatbot.chat_multimodal([image], self.PROMPT)


class Summarizer:
    PROMPT = (
        "Given a series of sentences where each sentence is a caption for an image, ",
        "please extract the most common object that exist in all sentences. ",
        "Output one noun-form object without any explanation and the template is: A photo of <object>",
        "Example: A photo of cat",
        "Do not include irrelevant or incorrect information. ",
        "The captions are as follows: \n",
    )

    def __init__(self, chatbot: Chatbot) -> None:
        self._chatbot = chatbot

    def __call__(self, captions: list[str]) -> list[str]:
        captions = [f'{i}. {caption}' for i, caption in enumerate(captions, 1)]
        prompt = ''.join(self.PROMPT) + '\n'.join(captions)
        print(prompt)
        return self._chatbot.chat(prompt)