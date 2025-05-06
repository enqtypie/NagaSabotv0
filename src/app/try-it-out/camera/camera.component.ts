import { Component, ViewChild, ElementRef, OnDestroy, PLATFORM_ID, Inject, AfterViewInit, Output, EventEmitter } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';

@Component({
  selector: 'app-camera',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './camera.component.html'
})
export class CameraComponent implements OnDestroy, AfterViewInit {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvas') canvas!: ElementRef<HTMLCanvasElement>;

  @Output() videoRecorded = new EventEmitter<Blob>();
  @Output() close = new EventEmitter<void>();

  showCameraModal = true;
  isRecording = false;
  isModalClosing = false;
  areLipsVisible = false;
  private mediaStream: MediaStream | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private faceLandmarker: any = null;
  private animationFrameId: number | null = null;
  private isViewInitialized = false;
  private noLipsDetectedCount = 0;
  private readonly NO_LIPS_THRESHOLD = 10;
  
  readonly REQUIRED_FRAMES = 75;
  currentFrameCount = 0;
  isFrameCollectionComplete = false;
  canvasWidth = 640;
  canvasHeight = 480;
  frameCount = 0;

  constructor(
    @Inject(PLATFORM_ID) private platformId: Object
  ) {}

  ngAfterViewInit() {
    if (isPlatformBrowser(this.platformId)) {
      this.isViewInitialized = true;
      this.initializeFaceLandmarker();
      this.openCameraModal();
    }
  }

  private async initializeFaceLandmarker() {
    if (!isPlatformBrowser(this.platformId)) return;

    try {
      const { FaceLandmarker, FilesetResolver, DrawingUtils } = await import('@mediapipe/tasks-vision');
      
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
      );
      
      this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
          delegate: "GPU"
        },
        outputFaceBlendshapes: false,
        runningMode: "VIDEO",
        numFaces: 1
      });

      (this as any).FaceLandmarker = FaceLandmarker;
      (this as any).DrawingUtils = DrawingUtils;
    } catch (error) {
      console.error('Error initializing face landmarker:', error);
    }
  }

  initializeCanvasSize() {
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

      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');

      this.mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          aspectRatio: { ideal: 4/3 },
          frameRate: { ideal: 30 },
          facingMode: 'user',
          deviceId: videoDevices[0]?.deviceId
        }
      });

      await this.waitForViewInitialization();
      
      if (!this.faceLandmarker) {
        await this.initializeFaceLandmarker();
      }
      
      this.videoElement.nativeElement.srcObject = this.mediaStream;
      
      await new Promise<void>((resolve) => {
        this.videoElement.nativeElement.onloadedmetadata = () => {
          this.videoElement.nativeElement.play();
          this.initializeCanvasSize();
          resolve();
          this.startFaceTracking();
        };
      });

      // Add event listeners for orientation changes and resize
      window.addEventListener('resize', this.handleResize);
      window.addEventListener('orientationchange', this.handleOrientationChange);

    } catch (err: any) {
      console.error('Camera error:', err);
      alert(this.getErrorMessage(err));
      this.closeCameraModal();
    }
  }

  private handleResize = () => {
    this.initializeCanvasSize();
  };

  private handleOrientationChange = () => {
    setTimeout(() => this.initializeCanvasSize(), 100);
  };

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
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    if (!this.faceLandmarker || !this.videoElement?.nativeElement || !this.canvas?.nativeElement) return;

    const ctx = this.canvas.nativeElement.getContext('2d');
    if (!ctx) return;

    const drawingUtils = new (this as any).DrawingUtils(ctx);
    const FaceLandmarker = (this as any).FaceLandmarker;
    
    const detectFaces = async () => {
      try {
        if (this.videoElement.nativeElement.paused || this.videoElement.nativeElement.ended) {
          await this.videoElement.nativeElement.play();
        }

        ctx.clearRect(0, 0, this.canvas.nativeElement.width, this.canvas.nativeElement.height);
        
        const results = this.faceLandmarker.detectForVideo(
          this.videoElement.nativeElement,
          performance.now()
        );

        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
          this.noLipsDetectedCount = 0;
          this.areLipsVisible = true;

          if (this.isRecording && !this.isFrameCollectionComplete) {
            this.currentFrameCount++;
            this.frameCount = this.currentFrameCount;
            if (this.currentFrameCount >= this.REQUIRED_FRAMES) {
              this.isFrameCollectionComplete = true;
              this.stopRecording();
              return;
            }
          }

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
          this.noLipsDetectedCount++;
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
      'NotAllowedError': 'Please make sure you have granted camera permissions.',
      'NotFoundError': 'No camera device found.',
      'NotReadableError': 'Camera is in use by another application.',
      'OverconstrainedError': 'Camera does not support the requested constraints.',
      'StreamApiNotSupportedError': 'Stream API is not supported in this browser.'
    };

    return `Unable to access camera. ${messages[error.name] || `Error: ${error.message || 'Unknown error occurred'}`}`;
  }

  closeCameraModal() {
    this.isModalClosing = true;
    setTimeout(() => {
      this.showCameraModal = false;
      this.isModalClosing = false;
      this.stopCamera();
      this.close.emit();
    }, 300);
  }

  startRecording() {
    if (!this.mediaStream) return;

    this.chunks = [];
    this.isRecording = true;
    this.currentFrameCount = 0;
    this.frameCount = 0;
    this.isFrameCollectionComplete = false;
    
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

    this.mediaRecorder.start();
  }

  stopRecording() {
    if (!this.isRecording) return;
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
    const videoBlob = new Blob(this.chunks, { type: 'video/webm' });
    if (videoBlob) {
      this.videoRecorded.emit(videoBlob);
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

  ngOnDestroy() {
    this.stopCamera();
    window.removeEventListener('resize', this.handleResize);
    window.removeEventListener('orientationchange', this.handleOrientationChange);
  }
} 