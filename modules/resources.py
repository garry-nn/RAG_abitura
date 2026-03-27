""" КЛАССы Ресурсов (артефактов)
Опишем классы ресурсов/артефактов, являющихся описаниями входных/выходных объектов задач домена.
"""

from typing import List, Union, Literal, Annotated, Dict, Type, Any
from annotated_types import MaxLen, Le, MinLen
from pydantic import BaseModel, Field
from datetime import date


class DocMetainfoRdf(BaseModel):
    """ Описание сущности документ в формате rdf для внесения в базу знаний """
    doc_id: Annotated[str, MaxLen(50)] = Field(..., description="идентификатор документа")
    title: str = Field(..., description="полное название документа")
    abstract: str = Field(..., description="краткое содержание документа")
    source: str = Field(..., description="адрес оригинала документа, URL - откуда взят документ")
    start_date: date = Field(..., description="дата регистрации документа в базе")


class MarkedTextClass(BaseModel):
    """ размеченный Markdown текст (документ или фрагмент) для анализа """
    text: str = Field(..., description="размеченный текст")


class ExtractedCQClass(BaseModel):
    """ извлеченный запрос """
    query: str = Field(..., description="текст запроса")
    answer: str = Field(..., description="текст ответа")
    citations: List[str] = Field(..., description="список цитат, предложений из текста без изменений, на основе которых сформирован ответ")


class CQListCLass(BaseModel):
    """ Список извлеченных CQ """
    extracted_cqs: List[ExtractedCQClass] = Field(..., description="список объектов ExtractedCQ")
