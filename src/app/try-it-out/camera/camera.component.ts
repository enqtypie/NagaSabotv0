import { Component, ViewChild, ElementRef, OnDestroy, PLATFORM_ID, Inject, AfterViewInit, Output, EventEmitter, Input } from '@angular/core';
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

  @Input() showHeadTiltWarning = false;
  @Input() showMouthMovementFeedback = false;

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
  canvasHeight = 640; // Changed to match 1:1 aspect ratio
  frameCount = 0;

  headTiltAngle: number | null = null;
  isMouthMoving = false;
  Math = Math; // Make Math available in template

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
      
      // Calculate the size for a 1:1 square format
      // First get the original dimensions
      let sourceWidth = video.videoWidth || 640;
      let sourceHeight = video.videoHeight || 480;
      
      // Determine the square size based on the smaller dimension
      const squareSize = Math.min(sourceWidth, sourceHeight);
      
      // Set the canvas dimensions to be square with device pixel ratio
      this.canvasWidth = squareSize * devicePixelRatio;
      this.canvasHeight = squareSize * devicePixelRatio;

      canvas.width = this.canvasWidth;
      canvas.height = this.canvasHeight;

      // Scale down the drawing context to counteract the pixel ratio scaling
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset any previous transforms
        ctx.scale(devicePixelRatio, devicePixelRatio);
      }

      // Set the CSS size to match the square display size
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      canvas.style.objectFit = 'cover';

      // Also set the video element to match 1:1
      video.style.width = '100%';
      video.style.height = '100%';
      video.style.objectFit = 'cover';
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

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('getUserMedia is not supported in this browser');
      }

      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');

      this.mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 640 }, // Changed to match 1:1 aspect ratio
          aspectRatio: { ideal: 1 }, // Changed to 1:1 aspect ratio
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
        
        // Calculate offsets for centering face in 1:1 square
        const videoWidth = this.videoElement.nativeElement.videoWidth;
        const videoHeight = this.videoElement.nativeElement.videoHeight;
        const squareSize = Math.min(videoWidth, videoHeight);
        
        // Get detection results from face landmarker
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
            // Create a mirrored and adjusted version of the landmarks for square aspect ratio
            const mirroredLandmarks = landmarks.map((point: { x: number; y: number; z: number }) => {
              // Mirror horizontally
              let x = 1 - point.x;
              
              // Adjust coordinates for square aspect ratio if needed
              if (videoWidth !== videoHeight) {
                if (videoWidth > videoHeight) {
                  // Landscape orientation - center horizontally
                  const offsetX = (videoWidth - squareSize) / (2 * videoWidth);
                  x = (1 - point.x - offsetX) * (videoWidth / squareSize);
                  if (x < 0) x = 0;
                  if (x > 1) x = 1;
                } else {
                  // Portrait orientation - center vertically
                  const offsetY = (videoHeight - squareSize) / (2 * videoHeight);
                  let y = (point.y - offsetY) * (videoHeight / squareSize);
                  if (y < 0) y = 0;
                  if (y > 1) y = 1;
                  return { x, y, z: point.z };
                }
              }
              
              return { x, y: point.y, z: point.z };
            });

            // Draw lip connectors with enhanced style
            drawingUtils.drawConnectors(
              mirroredLandmarks,
              FaceLandmarker.FACE_LANDMARKS_LIPS,
              { color: "#FFFFFF", lineWidth: 2 }
            );

            // Draw lip points with enhanced style
            const lipPoints = FaceLandmarker.FACE_LANDMARKS_LIPS.flat();
            for (let i = 0; i < lipPoints.length; i += 2) {
              const index = lipPoints[i];
              const point = mirroredLandmarks[index];
              if (!point) continue;
              ctx.beginPath();
              ctx.arc(
                point.x * this.canvas.nativeElement.width,
                point.y * this.canvas.nativeElement.height,
                3, // Slightly larger points
                0,
                2 * Math.PI
              );
              ctx.fillStyle = '#FF3030';
              ctx.fill();
              
              // Add glow effect
              ctx.beginPath();
              ctx.arc(
                point.x * this.canvas.nativeElement.width,
                point.y * this.canvas.nativeElement.height,
                5,
                0,
                2 * Math.PI
              );
              ctx.fillStyle = 'rgba(255, 48, 48, 0.3)';
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
    
    // Configure media recorder for square video
    const videoTrack = this.mediaStream.getVideoTracks()[0];
    const videoSettings = videoTrack.getSettings();
    
    // Create a square video with equal width and height
    const squareSize = Math.min(videoSettings.width || 640, videoSettings.height || 640);
    
    // Create a new canvas for video recording in 1:1 format
    const recordCanvas = document.createElement('canvas');
    recordCanvas.width = squareSize;
    recordCanvas.height = squareSize;
    const recordCtx = recordCanvas.getContext('2d');
    
    // Create a MediaStream from the canvas
    const recordStream = recordCanvas.captureStream(30); // 30fps
    
    // Add audio track if available
    const audioTracks = this.mediaStream.getAudioTracks();
    if (audioTracks.length > 0) {
      recordStream.addTrack(audioTracks[0]);
    }
    
    // Initialize MediaRecorder with optimal settings
    this.mediaRecorder = new MediaRecorder(recordStream, {
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

    // Start recording process
    this.mediaRecorder.start();
    
    // Start drawing video frames to the recording canvas
    const drawVideoFrame = () => {
      if (!this.isRecording) return;
      
      const video = this.videoElement.nativeElement;
      
      // Calculate source and destination parameters for a centered square crop
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;
      
      let sourceX = 0;
      let sourceY = 0;
      let sourceSize = Math.min(videoWidth, videoHeight);
      
      if (videoWidth > videoHeight) {
        // Landscape video - center horizontally
        sourceX = (videoWidth - sourceSize) / 2;
      } else {
        // Portrait video - center vertically
        sourceY = (videoHeight - sourceSize) / 2;
      }
      
      // Draw the video frame on the recording canvas (cropped to square)
      if (recordCtx) {
        recordCtx.drawImage(
          video,
          sourceX, sourceY, sourceSize, sourceSize,
          0, 0, squareSize, squareSize
        );
      }
      
      if (this.isRecording) {
        requestAnimationFrame(drawVideoFrame);
      }
    };
    
    drawVideoFrame();
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

  private async startFaceDetection() {
    // This is a placeholder for face detection logic
    // In a real implementation, you would use a face detection library
    // like MediaPipe or TensorFlow.js to detect face landmarks
    
    // For demo purposes, we'll simulate random head tilt and mouth movement
    setInterval(() => {
      if (this.showHeadTiltWarning) {
        this.headTiltAngle = Math.random() * 40 - 20; // Random angle between -20 and 20
      }
      if (this.showMouthMovementFeedback) {
        this.isMouthMoving = Math.random() > 0.5;
      }
    }, 1000);
  }
}