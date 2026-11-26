/** JSON crudo devuelto por POST /classify (backend FastAPI). */
export interface ClassifyApiResponse {
  error?: boolean;
  categoria?: string;
  /** Probabilidad en escala 0-1. */
  confianza?: number;
  emoji?: string;
  recomendacion?: string;
  /** Mapa categoría -> probabilidad (0-1). */
  todas_las_predicciones?: Record<string, number>;
  modelo_afinado?: boolean;
  mensaje?: string | null;
}

export interface ClassBreakdownItem {
  label: string;
  /** Porcentaje 0-100. */
  percent: number;
}

/** Resultado normalizado que consume la interfaz. */
export interface ClassificationResponse {
  prediction: string;
  /** Confianza en escala 0-1 (siempre finita). */
  confidence: number;
  emoji: string;
  disposal_recommendation?: string;
  breakdown: ClassBreakdownItem[];
}
