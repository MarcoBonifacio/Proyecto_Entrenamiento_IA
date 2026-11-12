export interface ClassificationResponse {
  prediction: string;
  confidence: number;
  disposal_recommendation?: string;
}
