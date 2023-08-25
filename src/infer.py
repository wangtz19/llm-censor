from model import BertCensor, ChatGLMAPICensor, ChatGPTCensor
import argparse

MODEL = {
    "bert": BertCensor,
    "chatglm": ChatGLMAPICensor,
    "chatgpt": ChatGPTCensor
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert", choices=MODEL.keys(), 
                        help="The model of the censor.")
    parser.add_argument("--model_checkpoint", type=str, default="hfl/chinese-bert-wwm", 
                        help="The checkpoint of bert model.")
    parser.add_argument("--tokenizer_checkpoint", type=str, default="hfl/chinese-bert-wwm",
                        help="The checkpoint of bert tokenizer.")
    parser.add_argument("--api_url", type=str, default="http://0.0.0.0:8000",
                        help="The url of the ChatGLM api.")
    args = parser.parse_args()

    censor = MODEL[args.model](model_checkpoint=args.model_checkpoint,
                                 tokenizer_checkpoint=args.tokenizer_checkpoint,
                                 api_url=args.api_url)
    while True:
        text = input("Input text: ")
        print(censor.predict(text))

if __name__ == "__main__":
    main()