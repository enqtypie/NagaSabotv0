import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TryItOutComponent } from './try-it-out.component';

describe('TryItOutComponent', () => {
  let component: TryItOutComponent;
  let fixture: ComponentFixture<TryItOutComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TryItOutComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(TryItOutComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
