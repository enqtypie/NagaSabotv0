// app.routes.ts
import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => 
      import('./homepage/homepage.component').then(m => m.HomeComponent)
  },
  {
    path: 'try-it-out',
    loadComponent: () => 
      import('./try-it-out/try-it-out.component').then(m => m.TryItOutComponent)
  }
];