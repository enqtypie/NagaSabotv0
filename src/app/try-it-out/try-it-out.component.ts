import { Component, ViewChild, ElementRef, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { VideoResultComponent, VideoResult } from '../video-result/video-result.component';
import { HeaderComponent } from '../header/header.component';
import { CameraComponent } from './camera/camera.component';
import { VideoService } from '../services/video.service';
import { HeadsUpDialogComponent } from './heads-up-dialog/heads-up-dialog.component';

@Component({
  selector: 'app-try-it-out',
  standalone: true,
  imports: [
    CommonModule,
    VideoResultComponent,
    HeaderComponent,
    CameraComponent,
    HeadsUpDialogComponent
  ],
  templateUrl: './try-it-out.component.html'
})
export class TryItOutComponent implements OnInit {
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  showCameraModal = false;
  videoBlob: Blob | null = null;
  isLoading = false;
  errorMessage: string | null = null;
  predictionResult: VideoResult | null = null;
  showHeadsUpDialog = true; // Always start with the dialog visible

  constructor(
    private videoService: VideoService
  ) {}

  ngOnInit() {
    // Force the dialog to show for testing - remove in production
    this.resetDialogAcknowledgment();
    
    // Instead of checking localStorage on init, we'll only rely on explicit user action
    // to close the dialog. This ensures it doesn't disappear unexpectedly.
    this.showHeadsUpDialog = true;
  }

  // For testing purposes - forces the dialog to appear
  resetDialogAcknowledgment() {
    localStorage.removeItem('hasAcknowledgedLipreadingDialog');
    console.log('Dialog acknowledgment has been reset');
    this.showHeadsUpDialog = true;
  }

  // Method to explicitly show the dialog
  showDialog() {
    this.showHeadsUpDialog = true;
  }

  onDialogClose() {
    // Only close the dialog when explicitly requested by user
    console.log('Dialog close requested');
    localStorage.setItem('hasAcknowledgedLipreadingDialog', 'true');
    this.showHeadsUpDialog = false;
  }

  openCameraModal() {
    // Always open the camera modal when requested, regardless of dialog state
    this.showCameraModal = true;
  }

  handleVideoRecorded(blob: Blob) {
    this.videoBlob = blob;
    this.isLoading = true;
    this.errorMessage = null;
    const file = new File([blob], 'recorded.webm', { type: 'video/webm' });
    this.handleVideoUpload(file);
    // Close the camera modal after recording
    this.showCameraModal = false;
  }

  handleFileSelection(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      this.videoBlob = input.files[0];
      this.isLoading = true;
      this.errorMessage = null;
      this.handleVideoUpload(input.files[0]);
    }
  }

  private handleVideoUpload(file: File) {
    this.isLoading = true;
    this.errorMessage = null;
    
    this.videoService.uploadVideo(file).subscribe({
      next: (result) => {
        this.predictionResult = result;
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error uploading video:', error);
        this.errorMessage = error.message || 'Error processing video. Please try again.';
        this.isLoading = false;
      }
    });
  }

  handleRestart() {
    if (this.videoBlob) {
      this.videoBlob = null;
    }
    if (this.fileInput) {
      this.fileInput.nativeElement.value = '';
    }
    this.predictionResult = null;
    this.errorMessage = null;
  }
}