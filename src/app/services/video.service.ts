import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map } from 'rxjs';
import { VideoResult, PredictionItem } from '../video-result/video-result.component';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class VideoService {
  private apiUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  uploadVideo(videoFile: File): Observable<VideoResult> {
    const formData = new FormData();
    formData.append('video', videoFile);

    return this.http.post<any>(`${this.apiUrl}/predict`, formData).pipe(
      map(response => {
        if (response.status === 'error') {
          throw new Error(response.message || 'Unknown error occurred');
        }

        // Create video URL for preview
        const videoUrl = URL.createObjectURL(videoFile);

        // Get the top prediction from the first item in top_predictions
        const topPrediction = response.top_predictions?.[0]?.phrase || 'No phrase detected';
        const topConfidence = response.top_predictions?.[0]?.confidence || 0;

        return {
          videoUrl,
          phrase: topPrediction,
          topPredictions: response.top_predictions || [],
          metrics: {
            confidence: topConfidence,
            open_mouth_ratio: response.metrics?.open_mouth_ratio || 0,
            frames_processed: response.metrics?.frames_processed || 0,
            processing_time: response.metrics?.processing_time || 0
          },
          timestamp: Date.now()
        };
      })
    );
  }
} 