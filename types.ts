import type * as THREE from 'three';

// Fix: Add an index signature to make the interface compatible with THREE.ShaderMaterial uniforms.
export interface ShaderUniforms {
  [key: string]: { value: any };
  uTime: { value: number };
  uMouse: { value: { x: number, y: number } };
  uCameraAspect: { value: number };
  uNoiseType: { value: number };
  uBigWavesElevation: { value: number };
  uBigWavesFrequency: { value: { x: number, y: number } };
  uBigWavesSpeed: { value: number };
  uSmallWavesElevation: { value: number };
  uSmallWavesFrequency: { value: number };
  uSmallWavesSpeed: { value: number };
  uSmallWavesIterations: { value: number };
  uColorOffset: { value: number };
  uColorMultiplier: { value: number };
  uDepthColor: { value: THREE.Color };
  uSurfaceColor: { value: THREE.Color };
  // New uniforms for ripple effect
  uDisplacementMap: { value: THREE.Texture | null };
  uRippleStrength: { value: number };
}
