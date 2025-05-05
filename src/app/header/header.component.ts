// header.component.ts
import { Component } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './header.component.html'
})
export class HeaderComponent {
  isMenuOpen = false;
  constructor(private router: Router) {}

  scrollToSection(sectionId: string): void {
    if (this.router.url !== '/' && this.router.url !== '/home') {
      this.router.navigate(['/'], { queryParams: { scrollTo: sectionId } });
      this.isMenuOpen = false;
      return;
    }
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
      });
    }
    this.isMenuOpen = false;
  }

  handleTitleClick(event: Event) {
    if (this.router.url === '/' || this.router.url === '/home') {
      event.preventDefault();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }
}