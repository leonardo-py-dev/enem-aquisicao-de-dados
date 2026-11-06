from enem_pipeline.collectors.base import BaseCollector
from enem_pipeline.collectors.inep import InepCollector
from enem_pipeline.collectors.ibge import IbgeCollector
from enem_pipeline.collectors.camara import CamaraCollector
from enem_pipeline.collectors.openalex import OpenAlexCollector

COLLECTORS_REGISTRY = {
    "inep": InepCollector,
    "ibge": IbgeCollector,
    "camara": CamaraCollector,
    "openalex": OpenAlexCollector,
}

__all__ = [
    "BaseCollector",
    "InepCollector",
    "IbgeCollector",
    "CamaraCollector",
    "OpenAlexCollector",
    "COLLECTORS_REGISTRY",
]
