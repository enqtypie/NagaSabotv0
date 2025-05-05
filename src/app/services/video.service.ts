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

    return this.http.post<any>(`${this.apiUrl}/upload`, formData).pipe(
      map(response => {
        if (response.status === 'error') {
          throw new Error(response.error || 'Unknown error occurred');
        }
        
        return {
          videoUrl: URL.createObjectURL(videoFile),
          phrase: response.top_prediction || 'No phrase detected',
          topPredictions: response.top_predictions || [],
          metrics: response.metrics || {
            confidence: 0,
            precision: 0,
            recall: 0,
            f1_score: 0,
            accuracy: 0
          },
          timestamp: Date.now()
        };
      })
    );
  }
} 