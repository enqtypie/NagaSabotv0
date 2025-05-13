import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { VideoService } from '../services/video.service';

export interface PredictionItem {
  phrase: string;
  confidence: number;
}

export interface Metrics {
  confidence: number;
  open_mouth_ratio: number;
  frames_processed: number;
  processing_time: number;
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
export class VideoResultComponent {
  @Input() videoBlob: Blob | null = null;
  @Input() predictionResult: VideoResult | null = null;
  @Input() isLoading = false;
  @Input() error: string | null = null;
  @Output() restart = new EventEmitter<void>();

  constructor(private videoService: VideoService) {}

  ngOnInit() {
    if (!this.predictionResult && this.videoBlob) {
      this.uploadVideo();
    }
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
    
    this.videoService.uploadVideo(file).subscribe({
      next: (response) => {
        this.predictionResult = response;
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error uploading video:', error);
        this.error = error.message || 'Error processing video. Please try again.';
        this.isLoading = false;
      }
    });
  }

  tryAgain() {
    if (this.predictionResult?.videoUrl) {
      URL.revokeObjectURL(this.predictionResult.videoUrl);
    }
    this.restart.emit();
  }

  get confidencePercentage(): string {
    return this.predictionResult?.metrics?.confidence 
      ? `${(this.predictionResult.metrics.confidence * 100).toFixed(1)}%`
      : 'N/A';
  }

  get openMouthRatioPercentage(): string {
    return this.predictionResult?.metrics?.open_mouth_ratio
      ? `${(this.predictionResult.metrics.open_mouth_ratio * 100).toFixed(1)}%`
      : 'N/A';
  }

  get framesProcessed(): number {
    return this.predictionResult?.metrics?.frames_processed || 0;
  }

  get processingTimeFormatted(): string {
    return this.predictionResult?.metrics?.processing_time
      ? `${this.predictionResult.metrics.processing_time.toFixed(2)}s`
      : 'N/A';
  }
} 