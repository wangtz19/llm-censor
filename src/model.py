from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict
import torch
import requests
import json
from utils import PROMPT, ANSWER_PATTERN


class BaseCensor(object):

    def __init__(self) -> None:
        pass

    def predict(self, text: str) -> dict:
        pass


class BertCensor(BaseCensor):

    def __init__(self, 
                 model_checkpoint: str,
                 tokenizer_checkpoint: str="hfl/chinese-bert-wwm",
                 **kwargs
                ) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

    def predict(self, text: str) -> Dict:
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True,
                                max_length=512, # 512 is the maximum length of BERT
                                padding="max_length").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits[0]
        pred = torch.argmax(logits).item()
        return {"label": pred, "reason": ""}
    

class ChatGLMAPICensor(BaseCensor):

    def __init__(self, 
                 api_url: str="http://0.0.0.0:8000",
                 prompt: str=PROMPT,
                 answer_pattern: str=ANSWER_PATTERN,
                 **kwargs
                ) -> None:
        self.api_url = api_url

    def create_chat_completion(self, text: str) -> str:
        url = f"{self.api_url}/v1/chat/completions"
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": text
                }
            ]
        }
        response = requests.post(url, json=data)
        response = json.loads(response.text)
        return response["choices"][0]["message"]["content"]

    def predict(self, text: str) -> Dict:
        resp = self.create_chat_completion(self.prompt.format(text=text))
        try:
            answer = self.answer_pattern.search(resp).group(1).strip()
            pred = 1 if answer == "是" else 0
        except:
            print(resp)
            pred = -1
        return {"label": pred, "reason": resp}
    

class ChatGPTCensor(BaseCensor):

    def __init__(self,
                 api_key_path: str,
                 prompt: str=PROMPT,
                 answer_pattern: str=ANSWER_PATTERN,
                 **kwargs
                ) -> None:
        import openai
        openai.api_key_path = api_key_path
        self.prompt = prompt
        self.answer_pattern = answer_pattern
        self.create_fn = openai.ChatCompletion.create

    def create_chat_completion(self, text: str) -> str:
        try:
            response = self.create_fn(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["\n\n"]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            return ""
        
    def predict(self, text: str) -> Dict:
        resp = self.create_chat_completion(self.prompt.format(text=text))
        try:
            answer = self.answer_pattern.search(resp).group(1).strip()
            pred = 1 if answer == "是" else 0
        except:
            print(resp)
            pred = -1
        return {"label": pred, "reason": resp}
