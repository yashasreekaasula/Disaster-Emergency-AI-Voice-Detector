from typing import Literal
from pydantic import BaseModel,Field
class Analysis(BaseModel):
    verdict:Literal['Distress','Normal','No signal']
    confidence:float|None=Field(default=None,ge=0,le=1)
    distress_score:float=Field(ge=0,le=1)
    emotion:dict[str,float]
    environment:dict[str,float]
    events:list[dict]
    words:list[dict]
    keywords:list[dict]
    transcript:str
    explanation:list[dict]
    warnings:list[str]
    start:float
    end:float
    method:str
    model_config={'extra':'allow'}
