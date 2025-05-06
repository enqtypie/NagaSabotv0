import { Component, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { VideoResultComponent } from '../video-result/video-result.component';
import { VideoService } from '../services/video.service';
import { HeaderComponent } from '../header/header.component';
import { CameraComponent } from './camera/camera.component';

@Component({
  selector: 'app-try-it-out',
  standalone: true,
  imports: [CommonModule, VideoResultComponent, HeaderComponent, CameraComponent],
  templateUrl: './try-it-out.component.html'
})
export class TryItOutComponent {
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  showCameraModal = false;
  videoBlob: Blob | null = null;
  isLoading = false;
  errorMessage: string | null = null;

  constructor(
    private videoService: VideoService
  ) {}

  openCameraModal() {
    this.showCameraModal = true;
  }

  handleVideoRecorded(blob: Blob) {
    this.videoBlob = blob;
    this.isLoading = true;
    this.errorMessage = null;
    
    // Create a properly named file with the correct MIME type
    const file = new File([blob], 'recorded.webm', { type: 'video/webm' });
    
    console.log('Uploading video file to backend:', file);
    console.log('File size:', file.size);
    
    this.videoService.uploadVideo(file).subscribe({
      next: (result) => {
        console.log('Upload successful:', result);
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error uploading video:', error);
        this.errorMessage = 'Error processing video. Please try again.';
        this.isLoading = false;
      }
    });
  }

  handleFileSelection(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      console.log('File selected:', input.files[0]);
      console.log('File size:', input.files[0].size);
      this.videoBlob = input.files[0];
      this.isLoading = true;
      this.errorMessage = null;
      this.handleVideoUpload(input.files[0]);
    }
  }

  private handleVideoUpload(file: File) {
    this.videoService.uploadVideo(file).subscribe({
      next: (result) => {
        console.log('Upload successful:', result);
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error uploading video:', error);
        this.errorMessage = 'Error processing video. Please try again.';
        this.isLoading = false;
      }
    });
  }

  handleRestart() {
    console.log('Closing video result...');
    if (this.videoBlob) {
      console.log('Cleaning up video blob');
      this.videoBlob = null;
    }
    if (this.fileInput) {
      this.fileInput.nativeElement.value = '';
    }
  }
}