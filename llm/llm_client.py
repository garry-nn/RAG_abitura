from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

MODEL_NAME = "GigaChat"
GIGACHAT_CREDENTIALS = "MDE5ZDUyODgtMzcyNC03MjQ5LTkwMDQtOGZhZjFlY2EwMzEwOjgxN2RkZTdiLTlhODktNDVjMS1iZGNiLWE5MDdmYzY3NjZlMw=="
GIGACHAT_VERIFY_SSL = False

_client = None


def get_client():
    global _client
    if _client is None:
        _client = GigaChat(
            credentials=GIGACHAT_CREDENTIALS,
            verify_ssl_certs=GIGACHAT_VERIFY_SSL,
            model=MODEL_NAME
        )
    return _client


def call_llm(prompt: str, temperature: float = 0.0, max_tokens: int = 700) -> str:
    client = get_client()

    request = Chat(
        messages=[
            Messages(role=MessagesRole.USER, content=prompt)
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        model=MODEL_NAME
    )

    response = client.chat(request)
    return response.choices[0].message.content.strip()