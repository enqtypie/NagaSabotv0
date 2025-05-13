import { Component, Output, EventEmitter, OnInit, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-heads-up-dialog',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './heads-up-dialog.component.html'
})
export class HeadsUpDialogComponent implements OnInit, AfterViewInit {
  @Output() close = new EventEmitter<void>();

  ngOnInit() {
    console.log('HeadsUpDialogComponent initialized');
    document.body.style.overflow = 'hidden'; // Prevent scrolling while dialog is open
  }

  ngAfterViewInit() {
    console.log('HeadsUpDialogComponent rendered');
    // Focus trap - keeping focus within the dialog
    setTimeout(() => {
      const closeButton = document.querySelector('button');
      if (closeButton) {
        closeButton.focus();
      }
    }, );
  }

  onClose() {
    console.log('Dialog close button clicked');
    document.body.style.overflow = ''; // Re-enable scrolling
    this.close.emit();
  }
} 