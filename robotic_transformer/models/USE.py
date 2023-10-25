# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class USEncoder(nn.Module):
    """Universal Sentence Encoder Encoder (USEncoder) is a PyTorch module that
    uses a pre-trained transformer model to encode sentences into fixed-size
    vectors.

    Args:
        model_name (str, optional): The name or path of the pre-trained model.
            Defaults to "sentence-transformers/paraphrase-MiniLM-L6-v2".
        hidden_size (int, optional): The dimension of the output embeddings.
            Defaults to 384.

    Attributes:
        tokenizer (transformers.AutoTokenizer): The tokenizer for the model.
        model (transformers.AutoModel): The pre-trained transformer model.
        hidden_size (int): The dimension of the output embeddings.

    Example usage:
    ```python
    encoder = USEncoder()
    sentence = "This is an example sentence."
    embeddings = encoder(sentence)
    ```

    The `USEncoder` class allows you to encode a sentence into a fixed-size
    vector representation using a pre-trained transformer model.
    """

    def __init__(
        self, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        hidden_size=384,
    ):
        """Initialize a new USEncoder instance.

        Args:
            model_name (str, optional): The name or path of the pre-trained model.
                Defaults to "sentence-transformers/paraphrase-MiniLM-L6-v2".
            hidden_size (int, optional): The dimension of the output embeddings.
                Defaults to 384.
        """
        super().__init__()
        # Load the pretrained Universal Sentence Encoder model and tokenizer
        # embeddings.shape == torch.Size([1, 384])
        self._hidden_size = hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # embeddings.shape == torch.Size([1, 768])
        # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')  # NOQA
        # model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        # There is a problem loading the following model weight schema
        # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # model = AutoModel.from_pretrained('bert-base-uncased')

    def forward(self, sentences):
        """Encode a list of sentences into fixed-size vectors.

        Args:
            sentences (str or List[str]): A single sentence or a list of sentences to encode.

        Returns:
            torch.Tensor: A tensor containing the encoded representations of the input sentences.
            The shape of the tensor is (batch_size, hidden_size).

        Example usage:
        ```python
        encoder = USEncoder()
        sentences = ["This is an example sentence.", "Another example sentence."]
        embeddings = encoder(sentences)
        ```
        """
        # Use a tokenizer to convert sentences into tokens
        tokens = self.tokenizer(
            sentences, padding=True,
            truncation=True, return_tensors="pt",
        )
        device = self.model.device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        # Get the vector representation of a sentence
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings


if __name__ == '__main__':
    # Sentences to be encoded
    sentences = ["Pick apple from top drawer and place on counter."]
    USE_model = USEncoder()
    embeddings = USE_model(sentences)
    # Print sentence vector shape
    print(embeddings.shape)
