from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    retrieval_type: str
    score: float
    source: str
    section: str
    content: str

    def to_dict(self):
        return {
            "retrieval_type": self.retrieval_type,
            "score": self.score,
            "source": self.source,
            "section": self.section,
            "content": self.content
        }