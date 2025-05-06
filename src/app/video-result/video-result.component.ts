import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { VideoService } from '../services/video.service';

export interface PredictionItem {
  phrase: string;
  confidence: number;
}

export interface Metrics {
  confidence: number;
  precision: number;
  recall: number;
  f1_score: number;
  accuracy: number;
}

export interface VideoResult {
  videoUrl: string;
  phrase: string;
  topPredictions: PredictionItem[];
  metrics: Metrics;
  timestamp: number;
}

@Component({
  selector: 'app-video-result',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './video-result.component.html'
})
export class VideoResultComponent implements OnInit {
  @Input() videoBlob!: Blob;
  @Output() restart = new EventEmitter<void>();
  
  result: VideoResult = {
    videoUrl: '',
    phrase: 'Processing video...',
    topPredictions: [],
    metrics: {
      confidence: 0,
      precision: 0,
      recall: 0,
      f1_score: 0,
      accuracy: 0
    },
    timestamp: Date.now()
  };

  isLoading = true;
  error: string | null = null;

  constructor(private videoService: VideoService) {}

  ngOnInit() {
    this.uploadVideo();
  }

  private uploadVideo() {
    if (!this.videoBlob) {
      this.error = 'No video data available';
      this.isLoading = false;
      return;
    }

    this.isLoading = true;
    this.error = null;
    
    const file = new File([this.videoBlob], 'recorded-video.webm', { type: 'video/webm' });
    console.log('Uploading video from result component:', file);
    console.log('File size:', file.size);
    
    // Upload directly to backend
    this.videoService.uploadVideo(file).subscribe({
      next: (response) => {
        console.log('Upload response:', response);
        this.result = {
          videoUrl: response.videoUrl,
          phrase: response.phrase,
          topPredictions: response.topPredictions,
          metrics: response.metrics,
          timestamp: response.timestamp
        };
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error uploading video:', error);
        this.error = 'Error processing video. Please try again.';
        this.isLoading = false;
      }
    });
  }

  tryAgain() {
    if (this.result.videoUrl) {
      URL.revokeObjectURL(this.result.videoUrl);
    }
    this.restart.emit();
  }

  get accuracyPercentage(): string {
    return `${(this.result.metrics.accuracy * 100).toFixed(1)}%`;
  }

  get confidencePercentage(): string {
    return `${(this.result.metrics.confidence * 100).toFixed(1)}%`;
  }

  get precisionPercentage(): string {
    return `${(this.result.metrics.precision * 100).toFixed(1)}%`;
  }

  get recallPercentage(): string {
    return `${(this.result.metrics.recall * 100).toFixed(1)}%`;
  }

  get f1ScorePercentage(): string {
    return `${(this.result.metrics.f1_score * 100).toFixed(1)}%`;
  }
} 