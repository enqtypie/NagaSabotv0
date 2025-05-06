import { Component, ViewChild, ElementRef, OnDestroy, PLATFORM_ID, Inject, AfterViewInit } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { VideoResultComponent } from '../video-result/video-result.component';
import { VideoService } from '../services/video.service';
import { HeaderComponent } from '../header/header.component';

@Component({
  selector: 'app-try-it-out',
  standalone: true,
  imports: [CommonModule, VideoResultComponent, HeaderComponent],
  templateUrl: './try-it-out.component.html'
})
export class TryItOutComponent implements OnDestroy, AfterViewInit {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;
  @ViewChild('canvas') canvas!: ElementRef<HTMLCanvasElement>;

  showCameraModal = false;
  isRecording = false;
  isModalClosing = false;
  videoBlob: Blob | null = null;
  areLipsVisible = false;
  private mediaStream: MediaStream | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private faceLandmarker: any = null;
  private animationFrameId: number | null = null;
  private isViewInitialized = false;
  private lastVideoTime = -1;
  private noLipsDetectedCount = 0;
  private readonly NO_LIPS_THRESHOLD = 10;
  
  readonly REQUIRED_FRAMES = 75;
  currentFrameCount = 0;
  isFrameCollectionComplete = false;
  canvasWidth = 640;
  canvasHeight = 480; // 4:3 ratio
  isLoading = false;
  errorMessage: string | null = null;
  private hasRequestedPermission = false;

  constructor(
    private videoService: VideoService,
    @Inject(PLATFORM_ID) private platformId: Object
  ) {}

  ngAfterViewInit() {
    if (isPlatformBrowser(this.platformId)) {
      this.isViewInitialized = true;
      this.initializeFaceLandmarker();
    }
  }

  private async initializeFaceLandmarker() {
    if (!isPlatformBrowser(this.platformId)) return;

    try {
      // Use dynamic import to only load the MediaPipe library when needed
      const { FaceLandmarker, FilesetResolver, DrawingUtils } = await import('@mediapipe/tasks-vision');
      
      // Configure the vision module with lighter options
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
      );
      
      // Create the face landmarker with optimized settings
      this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
          delegate: "GPU" // Try to use GPU if available for better performance
        },
        outputFaceBlendshapes: false, // Disable if not needed
        runningMode: "VIDEO",
        numFaces: 1 // Only track one face to reduce computation
      });

      (this as any).FaceLandmarker = FaceLandmarker;
      (this as any).DrawingUtils = DrawingUtils;
    } catch (error) {
      console.error('Error initializing face landmarker:', error);
    }
  }

  private initializeCanvas() {
    if (this.videoElement && this.videoElement.nativeElement && this.canvas && this.canvas.nativeElement) {
      const video = this.videoElement.nativeElement;
      const canvas = this.canvas.nativeElement;
      const devicePixelRatio = window.devicePixelRatio || 1;

      // Always use 4:3 ratio for canvas
      let width = video.videoWidth || 640;
      let height = video.videoHeight || 480;

      // Force 4:3 ratio
      if (width / height > 4 / 3) {
        width = height * 4 / 3;
      } else {
        height = width * 3 / 4;
      }

      this.canvasWidth = width * devicePixelRatio;
      this.canvasHeight = height * devicePixelRatio;

      canvas.width = this.canvasWidth;
      canvas.height = this.canvasHeight;

      // Scale down the drawing context to counteract the pixel ratio scaling
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset any previous transforms
        ctx.scale(devicePixelRatio, devicePixelRatio);
      }

      // Set the CSS size to match the 4:3 display size
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      canvas.style.objectFit = 'contain';

      // Also set the video element to match 4:3
      video.style.width = '100%';
      video.style.height = '100%';
      video.style.objectFit = 'contain';
    }
  }

  async openCameraModal() {
    if (!isPlatformBrowser(this.platformId)) {
      console.error('Camera access is only available in browser environments');
      return;
    }

    // Reset all relevant state for a new session
    this.currentFrameCount = 0;
    this.isFrameCollectionComplete = false;
    this.areLipsVisible = false;
    this.noLipsDetectedCount = 0;
    this.lastVideoTime = -1;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    
    // Clear faceLandmarker to free memory
    if (this.faceLandmarker) {
      this.faceLandmarker.close();
      this.faceLandmarker = null;
    }

    this.showCameraModal = true;

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('getUserMedia is not supported in this browser');
      }

      // First, check if we already have permission
      const devices = await navigator.mediaDevices.enumerateDevices();
      const hasPermission = devices.some(device => device.kind === 'videoinput' && device.label !== '');

      if (!hasPermission && !this.hasRequestedPermission) {
        // Request permission first with minimal constraints
        await navigator.mediaDevices.getUserMedia({ 
          video: true 
        }).then(stream => {
          // Stop the temporary stream
          stream.getTracks().forEach(track => track.stop());
          this.hasRequestedPermission = true;
        }).catch(err => {
          throw new Error('Camera permission denied');
        });
      }

      // Now request with our desired constraints
      this.mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 }, // 4:3 ratio
          aspectRatio: { ideal: 4/3 },
          frameRate: { ideal: 30 },
          facingMode: 'user'
        }
      });

      await this.waitForViewInitialization();
      
      // Initialize canvas after video element is ready
      this.initializeCanvas();
      
      // Lazy load the faceLandmarker only when needed
      if (!this.faceLandmarker) {
        await this.initializeFaceLandmarker();
      }
      
      this.videoElement.nativeElement.srcObject = this.mediaStream;
      
      await new Promise<void>((resolve) => {
        this.videoElement.nativeElement.onloadedmetadata = () => {
          this.videoElement.nativeElement.play();
          this.initializeCanvas(); // Re-initialize canvas with actual video dimensions
          resolve();
          this.startFaceTracking();
        };
      });

      // Add event listeners for orientation changes
      window.addEventListener('orientationchange', () => {
        setTimeout(() => this.initializeCanvas(), 100);
      });
      
      // Add event listener for window resize
      window.addEventListener('resize', () => {
        this.initializeCanvas();
      });

    } catch (err: any) {
      console.error('Camera error:', err);
      alert(this.getErrorMessage(err));
      this.closeCameraModal();
    }
  }

  private async waitForViewInitialization() {
    return new Promise<void>(resolve => {
      const checkView = () => {
        if (this.isViewInitialized && this.videoElement && this.canvas) {
          resolve();
        } else {
          setTimeout(checkView, 100);
        }
      };
      checkView();
    });
  }

  private async startFaceTracking() {
    // Cancel any previous animation frame loop
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    if (!this.faceLandmarker || !this.videoElement?.nativeElement || !this.canvas?.nativeElement) return;

    const ctx = this.canvas.nativeElement.getContext('2d');
    if (!ctx) return;

    const drawingUtils = new (this as any).DrawingUtils(ctx);
    const FaceLandmarker = (this as any).FaceLandmarker;
    
    let lastProcessTime = 0;
    
    const detectFaces = async () => {
      try {
        if (this.videoElement.nativeElement.paused || this.videoElement.nativeElement.ended) {
          await this.videoElement.nativeElement.play();
        }

        // Clear previous drawing before detection
        ctx.clearRect(0, 0, this.canvas.nativeElement.width, this.canvas.nativeElement.height);
        
        const results = this.faceLandmarker.detectForVideo(
          this.videoElement.nativeElement,
          performance.now()
        );

        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
          // Reset the no lips detected counter when lips are found
          this.noLipsDetectedCount = 0;
          this.areLipsVisible = true;

          // Increment frame count if recording and lips are visible
          if (this.isRecording && !this.isFrameCollectionComplete) {
            this.currentFrameCount++;
            if (this.currentFrameCount >= this.REQUIRED_FRAMES) {
              this.isFrameCollectionComplete = true;
              this.stopRecording();
              return; // Exit the detection loop after stopping
            }
          }

          // Only draw the first face to save resources
          const landmarks = results.faceLandmarks[0];
          if (landmarks) {
            // Create a mirrored version of the landmarks
            const mirroredLandmarks = landmarks.map((point: { x: number; y: number; z: number }) => ({
              ...point,
              x: 1 - point.x // Mirror the x coordinate
            }));

            // Draw only the lip connectors with mirrored landmarks
            drawingUtils.drawConnectors(
              mirroredLandmarks,
              FaceLandmarker.FACE_LANDMARKS_LIPS,
              { color: "#FFFFFF", lineWidth: 2 }
            );

            // Draw lip points with mirrored landmarks
            const lipPoints = FaceLandmarker.FACE_LANDMARKS_LIPS.flat();
            // Draw only every other lip point to reduce rendering load
            for (let i = 0; i < lipPoints.length; i += 2) {
              const index = lipPoints[i];
              const point = mirroredLandmarks[index];
              if (!point) continue; // Skip if undefined
              ctx.beginPath();
              ctx.arc(
                point.x * this.canvas.nativeElement.width,
                point.y * this.canvas.nativeElement.height,
                2,
                0,
                2 * Math.PI
              );
              ctx.fillStyle = '#FF3030';
              ctx.fill();
            }
          }
        } else {
          // Increment the counter when no lips are detected
          this.noLipsDetectedCount++;
          
          // If we haven't detected lips for several consecutive frames, consider them not visible
          if (this.noLipsDetectedCount >= this.NO_LIPS_THRESHOLD) {
            this.areLipsVisible = false;
          }
        }
      } catch (error) {
        console.error('Face tracking error:', error);
      }

      if (!this.isFrameCollectionComplete) {
        this.animationFrameId = requestAnimationFrame(detectFaces);
      }
    };

    detectFaces();
  }

  private getErrorMessage(error: any): string {
    const messages: { [key: string]: string } = {
      'NotAllowedError': 'Camera access was denied. Please allow camera access in your browser settings and try again.',
      'NotFoundError': 'No camera device found. Please ensure your device has a camera.',
      'NotReadableError': 'Camera is in use by another application. Please close other apps using the camera.',
      'OverconstrainedError': 'Your camera does not support the required settings. Please try using a different camera.',
      'StreamApiNotSupportedError': 'Your browser does not support camera access. Please try a different browser.',
      'SecurityError': 'Camera access is not allowed in this context. Please ensure you are using HTTPS or localhost.',
      'AbortError': 'Camera access was aborted. Please try again.',
      'TypeError': 'Camera access failed. Please check your browser settings and try again.'
    };

    return `Unable to access camera. ${messages[error.name] || `Error: ${error.message || 'Unknown error occurred'}`}`;
  }

  closeCameraModal() {
    this.isModalClosing = true;
    setTimeout(() => {
      this.showCameraModal = false;
      this.isModalClosing = false;
      this.stopCamera();
      window.removeEventListener('orientationchange', () => this.initializeCanvas());
      window.removeEventListener('resize', () => this.initializeCanvas());
    }, 300);
  }

  startRecording() {
    if (!this.mediaStream) return;

    this.chunks = [];
    this.isRecording = true;
    this.currentFrameCount = 0;
    this.isFrameCollectionComplete = false;
    
    // Configure MediaRecorder with quality settings and 4:3 ratio
    this.mediaRecorder = new MediaRecorder(this.mediaStream, {
      mimeType: 'video/webm;codecs=vp9',
      videoBitsPerSecond: 2500000 // 2.5 Mbps
    });

    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        this.chunks.push(event.data);
      }
    };

    this.mediaRecorder.onstop = () => {
      this.handleRecordingComplete();
    };

    // Start MediaRecorder
    this.mediaRecorder.start();
  }

  stopRecording() {
    if (!this.isRecording) return; // Guard: only stop once
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
      this.isRecording = false;
      console.log('Recording stopped, waiting for blob...');
      if (this.showCameraModal) {
        this.closeCameraModal();
      }
    }
  }

  private handleRecordingComplete() {
    this.videoBlob = new Blob(this.chunks, { type: 'video/webm' });
    
    if (this.videoBlob) {
      this.isLoading = true;
      this.errorMessage = null;
      
      // Create a properly named file with the correct MIME type
      const file = new File([this.videoBlob], 'recorded.webm', { type: 'video/webm' });
      
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
  }

  stopCamera() {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    if (this.videoElement?.nativeElement) {
      this.videoElement.nativeElement.srcObject = null;
    }

    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    if (this.faceLandmarker) {
      this.faceLandmarker = null;
    }
  }

  handleFileSelection(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      console.log('File selected:', input.files[0]);
      console.log('File size:', input.files[0].size);
      this.videoBlob = input.files[0];
      this.isLoading = true;
      this.errorMessage = null;
      this.handleRecordingComplete();
    }
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
    
    // Reset states if needed, but don't reopen the camera
    this.currentFrameCount = 0;
    this.isFrameCollectionComplete = false;
  }

  ngOnDestroy() {
    this.stopCamera();
  }
}