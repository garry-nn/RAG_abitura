import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class Embedder:
    def __init__(self, model_name="intfloat/multilingual-e5-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embedding(self, text: str):
        # важно для e5
        if not text.startswith("passage:") and not text.startswith("query:"):
            text = "passage: " + text

        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded)

        emb = self.mean_pooling(model_output, encoded["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)

        return emb[0].cpu().numpy()