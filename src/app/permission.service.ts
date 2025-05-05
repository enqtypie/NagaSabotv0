// permission.service.ts
import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class PermissionService {
  private cameraPermission = new BehaviorSubject<boolean>(false);
  private storagePermission = new BehaviorSubject<boolean>(false);

  cameraPermission$ = this.cameraPermission.asObservable();
  storagePermission$ = this.storagePermission.asObservable();

  async requestCameraPermission(): Promise<boolean> {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      stream.getTracks().forEach(track => track.stop());
      this.cameraPermission.next(true);
      return true;
    } catch (err) {
      console.error('Camera permission denied:', err);
      this.cameraPermission.next(false);
      return false;
    }
  }

  async requestStoragePermission(): Promise<boolean> {
    try {
      const [fileHandle] = await (window as any).showOpenFilePicker({
        types: [
          {
            description: 'Videos',
            accept: {
              'video/*': ['.mp4', '.webm', '.ogg']
            }
          }
        ],
        multiple: false
      });
      this.storagePermission.next(true);
      return true;
    } catch (err) {
      console.error('Storage permission denied:', err);
      this.storagePermission.next(false);
      return false;
    }
  }

  hasCameraPermission(): boolean {
    return this.cameraPermission.value;
  }

  hasStoragePermission(): boolean {
    return this.storagePermission.value;
  }
}