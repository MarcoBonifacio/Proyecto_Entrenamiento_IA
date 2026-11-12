import { apiClient } from '../../../services/apiConfig';
import type { ClassificationResponse } from '../types';

export const classifyImageFile = async (file: File): Promise<ClassificationResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post<ClassificationResponse>('/classify', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  return response.data;
};
