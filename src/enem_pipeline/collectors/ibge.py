import datetime
from typing import Optional
from sqlalchemy.orm import Session
from loguru import logger
from enem_pipeline.collectors.base import BaseCollector


class IbgeCollector(BaseCollector):
    """Coletor para estatísticas, notícias socioeconômicas e indicadores do IBGE."""

    def __init__(self):
        super().__init__(source_name="ibge")
        self.noticias_url = "https://servicodados.ibge.gov.br/api/v3/noticias/"

    def _parse_ibge_date(self, date_str: str) -> Optional[datetime.datetime]:
        """Converte formato 'DD/MM/YYYY HH:MM:SS' para objeto datetime."""
        if not date_str:
            return None
        try:
            return datetime.datetime.strptime(date_str.strip(), "%d/%m/%Y %H:%M:%S")
        except Exception:
            try:
                return datetime.datetime.strptime(date_str.strip()[:10], "%d/%m/%Y")
            except Exception:
                return None

    def collect(self, db: Session, max_items: int = 50) -> any:
        run = self.start_run(db)
        source = self.get_or_create_source(db)
        run_status = "success"

        # 1. Coleta Notícias e Releases sobre temas sociais e econômicos
        try:
            params = {
                "qtd": max_items,
            }
            logger.info(f"[IBGE] Consultando API de Notícias: {self.noticias_url}")
            data = self.fetch_json(self.noticias_url, params=params)

            items = data.get("items", []) if isinstance(data, dict) else []
            logger.info(f"[IBGE] {len(items)} notícias e releases recebidos.")

            for item in items:
                ext_id = str(item.get("id"))
                title = item.get("titulo")
                intro = item.get("introducao", "")
                link = item.get("link", "")
                editorias = item.get("editorias", "")
                tipo = item.get("tipo", "Notícia")
                pub_date = self._parse_ibge_date(item.get("data_publicacao", ""))

                content = (
                    f"IBGE - {tipo.upper()}: {title}\n\n"
                    f"{intro}\n\n"
                    f"Editorias / Temas: {editorias}\n"
                    f"Link Oficial: {link}"
                )

                metadata = {
                    "tipo": tipo,
                    "editorias": editorias,
                    "produto_id": item.get("produto_id"),
                    "link": link,
                }

                self.save_document(
                    db=db,
                    run=run,
                    source=source,
                    external_id=f"IBGE_{ext_id}",
                    title=f"[IBGE] {title}",
                    original_content=content,
                    url=link,
                    language="pt",
                    published_at=pub_date,
                    metadata=metadata,
                )
        except Exception as exc:
            logger.warning(f"[IBGE] Falha ao coletar notícias online ({exc}). Prosseguindo com indicadores...")
            run_status = "partial"

        # 2. Ingestão de Indicadores Demográficos e Sociais de Referência (IBGE)
        try:
            indicadores_referencia = [
                {
                    "name": "População Residente no Brasil (Censo 2022)",
                    "value": 203080756,
                    "unit": "habitantes",
                    "description": "Total de pessoas residentes no território brasileiro apurado pelo Censo Demográfico 2022.",
                    "reference_period": "2022",
                    "geographic_location": "Brasil",
                    "methodology": "Censo Demográfico 2022 - IBGE",
                },
                {
                    "name": "Taxa de Analfabetismo de Pessoas de 15 Anos ou Mais",
                    "value": 5.4,
                    "unit": "%",
                    "description": "Percentual de pessoas com 15 anos ou mais que não sabem ler e escrever.",
                    "reference_period": "2023",
                    "geographic_location": "Brasil",
                    "methodology": "PNAD Contínua - Educação 2023 - IBGE",
                },
                {
                    "name": "Proporção de Pessoas Abaixo da Linha de Pobreza",
                    "value": 27.5,
                    "unit": "%",
                    "description": "População vivendo com menos de US$ 6,85 por dia (critério Banco Mundial).",
                    "reference_period": "2023",
                    "geographic_location": "Brasil",
                    "methodology": "Síntese de Indicadores Sociais (SIS) - IBGE",
                },
                {
                    "name": "Índice de Envelhecimento Populacional",
                    "value": 55.2,
                    "unit": "idosos por 100 jovens",
                    "description": "Número de pessoas de 65 anos ou mais para cada 100 pessoas de 0 a 14 anos.",
                    "reference_period": "2022",
                    "geographic_location": "Brasil",
                    "methodology": "Censo Demográfico 2022 - IBGE",
                },
            ]

            for ind in indicadores_referencia:
                self.save_indicator(
                    db=db,
                    run=run,
                    source=source,
                    name=ind["name"],
                    value=ind["value"],
                    unit=ind["unit"],
                    description=ind["description"],
                    reference_period=ind["reference_period"],
                    geographic_location=ind["geographic_location"],
                    methodology=ind["methodology"],
                )

            db.commit()
            self.finish_run(db, run, status=run_status)
        except Exception as exc:
            db.rollback()
            self.finish_run(db, run, status="failed", error_details=str(exc))
            raise

        return run
