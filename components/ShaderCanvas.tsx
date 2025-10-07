import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
// Fix: Import Controller type from lil-gui.
import GUI, { Controller } from 'lil-gui';
import type { ShaderUniforms } from '../types';

// GLSL Shaders
const vertexShader = `
  precision mediump float;

  uniform float uTime;
  uniform vec2 uMouse;
  uniform float uCameraAspect;
  uniform int uNoiseType; // 0: Perlin, 1: Simplex, 2: Worley, 3: FBM, 4: Smooth

  uniform float uBigWavesElevation;
  uniform vec2 uBigWavesFrequency;
  uniform float uBigWavesSpeed;
  uniform float uSmallWavesElevation;
  uniform float uSmallWavesFrequency;
  uniform float uSmallWavesSpeed;
  uniform float uSmallWavesIterations;

  // New uniforms for fluid displacement
  uniform sampler2D uDisplacementMap;
  uniform float uRippleStrength;

  // Classic Perlin 3D Noise (from original)
  vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
  vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
  vec3 fade(vec3 t) {return t*t*t*(t*(t*6.0-15.0)+10.0);}

  float cnoise(vec3 P){
    vec3 Pi0 = floor(P);
    vec3 Pi1 = Pi0 + vec3(1.0);
    Pi0 = mod(Pi0, 289.0);
    Pi1 = mod(Pi1, 289.0);
    vec3 Pf0 = fract(P);
    vec3 Pf1 = Pf0 - vec3(1.0);
    vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    vec4 iy = vec4(Pi0.yy, Pi1.yy);
    vec4 iz0 = Pi0.zzzz;
    vec4 iz1 = Pi1.zzzz;
    vec4 ixy = permute(permute(ix) + iy);
    vec4 ixy0 = permute(ixy + iz0);
    vec4 ixy1 = permute(ixy + iz1);
    vec4 gx0 = ixy0 / 7.0;
    vec4 gy0 = fract(floor(gx0) / 7.0) - 0.5;
    gx0 = fract(gx0);
    vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
    vec4 sz0 = step(gz0, vec4(0.0));
    gx0 -= sz0 * (step(0.0, gx0) - 0.5);
    gy0 -= sz0 * (step(0.0, gy0) - 0.5);
    vec4 gx1 = ixy1 / 7.0;
    vec4 gy1 = fract(floor(gx1) / 7.0) - 0.5;
    gx1 = fract(gx1);
    vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
    vec4 sz1 = step(gz1, vec4(0.0));
    gx1 -= sz1 * (step(0.0, gx1) - 0.5);
    gy1 -= sz1 * (step(0.0, gy1) - 0.5);
    vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
    vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
    vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
    vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
    vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
    vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
    vec3 g011 = vec3(gx1.z,gy1.z,gz1.x);
    vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);
    vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
    g000 *= norm0.x;
    g010 *= norm0.y;
    g100 *= norm0.z;
    g110 *= norm0.w;
    vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
    g001 *= norm1.x;
    g011 *= norm1.y;
    g101 *= norm1.z;
    g111 *= norm1.w;
    float n000 = dot(g000, Pf0);
    float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
    float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
    float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
    float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
    float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
    float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
    float n111 = dot(g111, Pf1);
    vec3 fade_xyz = fade(Pf0);
    vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
    vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
    float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
    return 2.2 * n_xyz;
  }
  
  // Simplex Noise
  float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod(i, 289.0);
    vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
  }

  // Fractional Brownian Motion using Simplex Noise
  float fbm(vec3 p) {
      float total = 0.0;
      float amplitude = 0.5;
      float frequency = 1.0;
      for (int i = 0; i < 6; i++) {
          total += snoise(p * frequency) * amplitude;
          frequency *= 2.0;
          amplitude *= 0.5;
      }
      return total;
  }

  // Worley Noise
  float worley(vec3 p) {
      vec3 p_int = floor(p);
      vec3 p_fract = fract(p);
      float min_dist = 1.0;
      for (int z = -1; z <= 1; z++) {
          for (int y = -1; y <= 1; y++) {
              for (int x = -1; x <= 1; x++) {
                  vec3 neighbor = vec3(float(x), float(y), float(z));
                  vec3 point = p_int + neighbor;
                  vec3 point_fract = fract(sin(point * 123.45 + 67.89) * 4321.0);
                  float dist = distance(p_fract, point_fract + neighbor);
                  min_dist = min(min_dist, dist);
              }
          }
      }
      return min_dist;
  }

  // Smooth Noise by Inigo Quilez
  vec3 hash( vec3 p ) {
    p = vec3( dot(p,vec3(127.1,311.7, 74.7)),
              dot(p,vec3(269.5,183.3,246.1)),
              dot(p,vec3(113.5,271.9,124.6)));
    return -1.0 + 2.0*fract(sin(p)*43758.5453123);
  }
  float smoothNoise(in vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f*f*(3.0-2.0*f);
    return mix( mix( mix( dot( hash( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                           dot( hash( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), u.x),
                      mix( dot( hash( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), 
                           dot( hash( i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) ), u.x), u.y),
                 mix( mix( dot( hash( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
                           dot( hash( i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) ), u.x),
                      mix( dot( hash( i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) ), 
                           dot( hash( i + vec3(1.0,1.0,1.0) ), f - vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
  }

  float noise(vec3 p) {
    if (uNoiseType == 0) return cnoise(p);
    if (uNoiseType == 1) return snoise(p);
    if (uNoiseType == 2) return 1.0 - worley(p) * 2.0;
    if (uNoiseType == 3) return fbm(p) * 2.0;
    if (uNoiseType == 4) return smoothNoise(p) * 2.0;
    return 0.0;
  }

  varying float vWaveElevation;

  void main() {
    vec4 modelPosition = modelMatrix * vec4(position, 1.0);

    // Big waves
    float waveElevation = noise(vec3(
        modelPosition.x * uBigWavesFrequency.x + uTime * uBigWavesSpeed,
        modelPosition.y * uBigWavesFrequency.y + uTime * uBigWavesSpeed,
        uTime * 0.1
    )) * uBigWavesElevation;

    // Small waves (FBM)
    for(float i = 1.0; i <= uSmallWavesIterations; i++) {
        waveElevation += snoise(vec3( // Use snoise for small waves for consistency
            modelPosition.x * uSmallWavesFrequency * i + uTime * uSmallWavesSpeed,
            modelPosition.y * uSmallWavesFrequency * i + uTime * uSmallWavesSpeed,
            uTime * 0.2
        )) * (uSmallWavesElevation / i);
    }
    
    vWaveElevation = waveElevation;
    
    // Fluid displacement from cursor
    float fluidDisplacement = texture2D(uDisplacementMap, uv).r;
    float totalElevation = waveElevation + fluidDisplacement * uRippleStrength;

    modelPosition.z = totalElevation;

    gl_Position = projectionMatrix * viewMatrix * modelPosition;
  }
`;

const fragmentShader = `
  precision mediump float;

  uniform vec3 uDepthColor;
  uniform vec3 uSurfaceColor;
  uniform float uColorOffset;
  uniform float uColorMultiplier;

  varying float vWaveElevation;

  void main() {
    float mixStrength = (vWaveElevation + uColorOffset) * uColorMultiplier;
    
    // Use smoothstep for a more gradual, "buttery" color transition
    float smoothedMix = smoothstep(0.0, 1.0, mixStrength);
    vec3 color = mix(uDepthColor, uSurfaceColor, smoothedMix);
    
    gl_FragColor = vec4(color, 1.0);
  }
`;

const rippleVertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const rippleFragmentShader = `
  uniform sampler2D uPrevTexture;
  uniform vec2 uMouse;
  uniform float uBrushSize;
  uniform float uDamping;
  uniform float uTrailPersistence;
  uniform vec2 uTexelSize;

  varying vec2 vUv;

  void main() {
    // Read previous state (height, prevHeight, trail)
    vec4 prevState = texture2D(uPrevTexture, vUv);
    float currentHeight = prevState.r;
    float prevHeight = prevState.g;
    float trail = prevState.b;

    // Propagate waves from neighbors
    vec2 north = vec2(vUv.x, vUv.y + uTexelSize.y);
    vec2 south = vec2(vUv.x, vUv.y - uTexelSize.y);
    vec2 east = vec2(vUv.x + uTexelSize.x, vUv.y);
    vec2 west = vec2(vUv.x - uTexelSize.x, vUv.y);

    float n = texture2D(uPrevTexture, north).r;
    float s = texture2D(uPrevTexture, south).r;
    float e = texture2D(uPrevTexture, east).r;
    float w = texture2D(uPrevTexture, west).r;

    float average = (n + s + e + w) * 0.5;
    float newHeight = average - prevHeight;
    
    // Add a new drop from the mouse to the trail
    float drop = max(0.0, 1.0 - distance(uMouse, vUv) / uBrushSize);
    drop = pow(drop, 3.0);
    
    // Update trail: add new drop and decay existing trail
    trail = max(trail, drop);
    trail *= uTrailPersistence;

    // Apply trail as a force to the wave height
    newHeight += trail * 0.05;

    // Apply damping to the whole simulation
    newHeight *= uDamping;

    // Store new height in red, current height in green, trail in blue
    gl_FragColor = vec4(newHeight, currentHeight, trail, 1.0);
  }
`;

// Post-processing shaders
const postVertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
  }
`;

const blurFragmentShader = `
  uniform sampler2D uTexture;
  uniform vec2 uDirection; // (1.0/width, 0.0) for H, (0.0, 1.0/height) for V
  varying vec2 vUv;

  void main() {
    vec4 color = vec4(0.0);
    // 5-tap Gaussian blur kernel
    float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    
    color += texture2D(uTexture, vUv) * weights[0];
    for(int i = 1; i < 5; i++) {
        color += texture2D(uTexture, vUv + float(i) * uDirection) * weights[i];
        color += texture2D(uTexture, vUv - float(i) * uDirection) * weights[i];
    }

    gl_FragColor = color;
  }
`;

const compositeFragmentShader = `
  uniform sampler2D uSceneTexture;
  uniform sampler2D uBlurTexture;
  uniform float uIntensity;
  uniform float uNoiseAlpha;
  uniform float uTime;
  varying vec2 vUv;

  // A simple pseudo-random noise function
  float rand(vec2 n) { 
    return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
  }

  void main() {
    vec4 sceneColor = texture2D(uSceneTexture, vUv);
    vec4 blurColor = texture2D(uBlurTexture, vUv);

    // Additive blending for a bloom/glow effect
    vec4 finalColor = sceneColor + blurColor * uIntensity;
    
    // Add animated film grain
    float noise = rand(vUv + fract(uTime));
    finalColor.rgb += (noise - 0.5) * uNoiseAlpha;

    gl_FragColor = finalColor;
  }
`;


const defaultPresets = {
  "Serene Twilight": {
    noiseType: 1, // Simplex
    uBigWavesElevation: 0.05, uBigWavesFrequency: { x: 0.6, y: 0.4 }, uBigWavesSpeed: 0.03,
    uSmallWavesElevation: 0.04, uSmallWavesFrequency: 2.5, uSmallWavesSpeed: 0.1, uSmallWavesIterations: 4.0,
    depthColor: '#0c3483', surfaceColor: '#a2b6df', uColorOffset: 0.2, uColorMultiplier: 3.0,
    blurIntensity: 0.8,
  },
  "Cherry Blossom": {
    noiseType: 1, // Simplex
    uBigWavesElevation: 0.04, uBigWavesFrequency: { x: 0.5, y: 0.4 }, uBigWavesSpeed: 0.03,
    uSmallWavesElevation: 0.06, uSmallWavesFrequency: 2.5, uSmallWavesSpeed: 0.12, uSmallWavesIterations: 4.0,
    depthColor: '#ff758c', surfaceColor: '#ffdde1', uColorOffset: 0.2, uColorMultiplier: 3.5,
    blurIntensity: 0.4,
  },
  "Warm Embrace": {
    noiseType: 1, // Simplex
    uBigWavesElevation: 0.03, uBigWavesFrequency: { x: 1.0, y: 0.8 }, uBigWavesSpeed: 0.06,
    uSmallWavesElevation: 0.05, uSmallWavesFrequency: 5.0, uSmallWavesSpeed: 0.2, uSmallWavesIterations: 4.0,
    depthColor: '#E85A4F', surfaceColor: '#FFF4E0', uColorOffset: 0.25, uColorMultiplier: 2.0,
    blurIntensity: 0.9,
  },
  "Golden Hour": {
    noiseType: 0, // Perlin
    uBigWavesElevation: 0.02, uBigWavesFrequency: { x: 0.5, y: 0.5 }, uBigWavesSpeed: 0.025,
    uSmallWavesElevation: 0.04, uSmallWavesFrequency: 4.0, uSmallWavesSpeed: 0.15, uSmallWavesIterations: 3.0,
    depthColor: '#F09819', surfaceColor: '#FFFDE4', uColorOffset: 0.2, uColorMultiplier: 2.0,
    blurIntensity: 0.7,
  },
  "Misty Meadow": {
    noiseType: 2, // Worley
    uBigWavesElevation: 0.08, uBigWavesFrequency: { x: 0.3, y: 0.3 }, uBigWavesSpeed: 0.02,
    uSmallWavesElevation: 0.04, uSmallWavesFrequency: 2.0, uSmallWavesSpeed: 0.05, uSmallWavesIterations: 3.0,
    depthColor: '#01416d', surfaceColor: '#48b1bf', uColorOffset: 0.15, uColorMultiplier: 3.0,
    blurIntensity: 1.2,
  },
  "Cloud Scape": {
    noiseType: 3, // FBM
    uBigWavesElevation: 0.15, uBigWavesFrequency: { x: 0.5, y: 0.4 }, uBigWavesSpeed: 0.03,
    uSmallWavesElevation: 0.0, uSmallWavesFrequency: 1.0, uSmallWavesSpeed: 0.0, uSmallWavesIterations: 1.0,
    depthColor: '#4a6e8a', surfaceColor: '#d3e0ea', uColorOffset: 0.2, uColorMultiplier: 3.0,
    blurIntensity: 1.5,
  },
  "Emerald Tide": {
    noiseType: 1, // Simplex
    uBigWavesElevation: 0.06, uBigWavesFrequency: { x: 0.4, y: 0.3 }, uBigWavesSpeed: 0.04,
    uSmallWavesElevation: 0.05, uSmallWavesFrequency: 3.0, uSmallWavesSpeed: 0.18, uSmallWavesIterations: 4.0,
    depthColor: '#00796B', surfaceColor: '#69F0AE', uColorOffset: 0.25, uColorMultiplier: 3.5,
    blurIntensity: 0.5,
  },
};
type Preset = typeof defaultPresets[keyof typeof defaultPresets];
type Presets = typeof defaultPresets;

const qualitySettings = {
    High: {
        pixelRatio: Math.min(window.devicePixelRatio, 2),
        geometrySegments: 256,
        antialias: true,
        fboResolution: 512,
        blurDownsample: 4,
    },
    Low: {
        pixelRatio: 1,
        geometrySegments: 128,
        antialias: false,
        fboResolution: 256,
        blurDownsample: 8,
    },
};
type Quality = keyof typeof qualitySettings;

const ShaderCanvas: React.FC = () => {
    const mountRef = useRef<HTMLDivElement>(null);
    const presetsRef = useRef<Presets>(JSON.parse(JSON.stringify(defaultPresets)));
    
    useEffect(() => {
        if (!mountRef.current) return;

        const isMobile = window.innerWidth < 768;
        const perfParams = { quality: (isMobile ? 'Low' : 'High') as Quality };
        let settings = qualitySettings[perfParams.quality];
        
        let lastAppliedPreset: Preset = presetsRef.current["Serene Twilight"];

        const currentMount = mountRef.current;
        const scene = new THREE.Scene();
        const sizes = { width: currentMount.clientWidth, height: currentMount.clientHeight };
        const camera = new THREE.PerspectiveCamera(75, sizes.width / sizes.height, 0.1, 100);
        camera.position.set(0, 0, 1.8);
        scene.add(camera);
        
        const renderer = new THREE.WebGLRenderer({ 
            antialias: settings.antialias, 
            alpha: true,
            powerPreference: 'high-performance'
        });
        renderer.setSize(sizes.width, sizes.height);
        renderer.setPixelRatio(settings.pixelRatio);
        currentMount.appendChild(renderer.domElement);

        // Fluid Simulation Setup
        const fluidScene = new THREE.Scene();
        const fluidCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        const fluidGeometry = new THREE.PlaneGeometry(2, 2);
        let fbo1 = new THREE.WebGLRenderTarget(settings.fboResolution, settings.fboResolution, { type: THREE.FloatType });
        let fbo2 = new THREE.WebGLRenderTarget(settings.fboResolution, settings.fboResolution, { type: THREE.FloatType });
        
        const fluidParams = { strength: 0.15, damping: 0.985, brushSize: 0.08, trail: 0.95 };

        const rippleUniforms = {
            uPrevTexture: { value: null },
            uMouse: { value: new THREE.Vector2(0.5, 0.5) },
            uBrushSize: { value: fluidParams.brushSize },
            uDamping: { value: fluidParams.damping },
            uTrailPersistence: { value: fluidParams.trail },
            uTexelSize: { value: new THREE.Vector2(1 / settings.fboResolution, 1 / settings.fboResolution) }
        };
        const rippleMaterial = new THREE.ShaderMaterial({
            vertexShader: rippleVertexShader,
            fragmentShader: rippleFragmentShader,
            uniforms: rippleUniforms
        });
        const fluidMesh = new THREE.Mesh(fluidGeometry, rippleMaterial);
        fluidScene.add(fluidMesh);

        const displacementParams = { type: 'Simplex' };
        const noiseTypes = { Perlin: 0, Simplex: 1, Worley: 2, FBM: 3, Smooth: 4 };
        
        const colorParams = {
            depthColor: '#d1e0ff',
            surfaceColor: '#fff4e0',
        };

        const uniforms: ShaderUniforms = {
            uTime: { value: 0 },
            uMouse: { value: { x: 9999, y: 9999 } },
            uCameraAspect: { value: camera.aspect },
            uNoiseType: { value: noiseTypes.Simplex },
            uBigWavesElevation: { value: 0.08 }, uBigWavesFrequency: { value: { x: 0.6, y: 0.4 } }, uBigWavesSpeed: { value: 0.04 },
            uSmallWavesElevation: { value: 0.06 }, uSmallWavesFrequency: { value: 2.0 }, uSmallWavesSpeed: { value: 0.15 }, uSmallWavesIterations: { value: 3.0 },
            uDepthColor: { value: new THREE.Color(colorParams.depthColor) },
            uSurfaceColor: { value: new THREE.Color(colorParams.surfaceColor) },
            uColorOffset: { value: 0.3 },
            uColorMultiplier: { value: 2.5 },
            uDisplacementMap: { value: null },
            uRippleStrength: { value: fluidParams.strength }
        };

        const material = new THREE.ShaderMaterial({ vertexShader, fragmentShader, uniforms });
        const geometry = new THREE.PlaneGeometry(4.5, 4.5, settings.geometrySegments, settings.geometrySegments);
        const mesh = new THREE.Mesh(geometry, material);
        mesh.rotation.x = -Math.PI * 0.35;
        scene.add(mesh);

        // Post-processing setup
        const postScene = new THREE.Scene();
        const postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        const postGeometry = new THREE.PlaneGeometry(2, 2);

        let sceneFBO = new THREE.WebGLRenderTarget(sizes.width, sizes.height, { type: THREE.FloatType });
        let blurFBO1 = new THREE.WebGLRenderTarget(sizes.width / settings.blurDownsample, sizes.height / settings.blurDownsample, { type: THREE.FloatType });
        let blurFBO2 = new THREE.WebGLRenderTarget(sizes.width / settings.blurDownsample, sizes.height / settings.blurDownsample, { type: THREE.FloatType });

        const blurHorizontalMaterial = new THREE.ShaderMaterial({
            vertexShader: postVertexShader,
            fragmentShader: blurFragmentShader,
            uniforms: {
                uTexture: { value: null },
                uDirection: { value: new THREE.Vector2(1 / (sizes.width / settings.blurDownsample), 0) }
            }
        });
        const blurVerticalMaterial = new THREE.ShaderMaterial({
            vertexShader: postVertexShader,
            fragmentShader: blurFragmentShader,
            uniforms: {
                uTexture: { value: null },
                uDirection: { value: new THREE.Vector2(0, 1 / (sizes.height / settings.blurDownsample)) }
            }
        });
        const compositeMaterial = new THREE.ShaderMaterial({
            vertexShader: postVertexShader,
            fragmentShader: compositeFragmentShader,
            uniforms: {
                uSceneTexture: { value: null },
                uBlurTexture: { value: null },
                uIntensity: { value: 1.0 },
                uNoiseAlpha: { value: 0.05 },
                uTime: { value: 0 }
            }
        });
        const postMesh = new THREE.Mesh(postGeometry, compositeMaterial);
        postScene.add(postMesh);
        
        const gui = new GUI();
        const controllers: { [key: string]: Controller } = {};
        
        const applyPreset = (preset: Preset) => {
            lastAppliedPreset = preset;
            const noiseTypeName = Object.keys(noiseTypes).find(key => noiseTypes[key as keyof typeof noiseTypes] === preset.noiseType) || 'Simplex';
            controllers.noiseType.setValue(noiseTypeName);
            controllers.bigWavesElevation.setValue(preset.uBigWavesElevation);
            controllers.bigWavesFrequencyX.setValue(preset.uBigWavesFrequency.x);
            controllers.bigWavesFrequencyY.setValue(preset.uBigWavesFrequency.y);
            controllers.bigWavesSpeed.setValue(preset.uBigWavesSpeed);
            controllers.smallWavesElevation.setValue(preset.uSmallWavesElevation);
            controllers.smallWavesFrequency.setValue(preset.uSmallWavesFrequency);
            controllers.smallWavesSpeed.setValue(preset.uSmallWavesSpeed);
            let iterations = preset.uSmallWavesIterations;
            if (perfParams.quality === 'Low') {
                iterations = Math.min(preset.uSmallWavesIterations, 2.0);
            }
            controllers.smallWavesIterations.setValue(iterations);
            controllers.colorOffset.setValue(preset.uColorOffset);
            controllers.colorMultiplier.setValue(preset.uColorMultiplier);
            controllers.depthColor.setValue(preset.depthColor);
            controllers.surfaceColor.setValue(preset.surfaceColor);
            controllers.blurIntensity.setValue(preset.blurIntensity);
        };

        let presetsFolder: GUI;
        const rebuildPresetsFolder = (currentPresets: Presets) => {
            if (presetsFolder) {
                presetsFolder.destroy();
            }
            presetsFolder = gui.addFolder('Presets');
            const presetFunctions = Object.fromEntries(Object.entries(currentPresets).map(([name, preset]) => [name, () => applyPreset(preset)]));
            for (const key in presetFunctions) {
                presetsFolder.add(presetFunctions, key);
            }
            presetsFolder.open();
        };
        
        const ioControls = {
            exportPresets: () => {
                const json = JSON.stringify(presetsRef.current, null, 2);
                const blob = new Blob([json], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'fluidum-presets.json';
                a.click();
                URL.revokeObjectURL(url);
            },
            importPresets: () => {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = '.json';
                input.onchange = (event) => {
                    const file = (event.target as HTMLInputElement).files?.[0];
                    if (!file) return;
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        try {
                            const newPresets = JSON.parse(e.target?.result as string);
                            if (typeof newPresets === 'object' && newPresets !== null) {
                                presetsRef.current = newPresets;
                                rebuildPresetsFolder(presetsRef.current);
                                const firstPreset = Object.values(presetsRef.current)[0] as Preset;
                                if (firstPreset) applyPreset(firstPreset);
                            } else {
                                alert('Invalid presets file format.');
                            }
                        } catch (error) {
                            alert('Error parsing presets file.');
                        }
                    };
                    reader.readAsText(file);
                };
                input.click();
            }
        };

        const perfFolder = gui.addFolder('Performance');
        perfFolder.add(perfParams, 'quality', ['High', 'Low']).name('Quality').onChange((value: Quality) => {
            settings = qualitySettings[value];
            renderer.setPixelRatio(settings.pixelRatio);
            fbo1.setSize(settings.fboResolution, settings.fboResolution);
            fbo2.setSize(settings.fboResolution, settings.fboResolution);
            rippleUniforms.uTexelSize.value.set(1 / settings.fboResolution, 1 / settings.fboResolution);
            // Re-apply preset to update iterations based on new quality
            applyPreset(lastAppliedPreset);
            // Manually trigger resize to update blur buffers
            handleResize();
        });
        perfFolder.open();

        gui.add(ioControls, 'importPresets').name('Import Presets');
        gui.add(ioControls, 'exportPresets').name('Export Presets');
        
        rebuildPresetsFolder(presetsRef.current);

        const blurFolder = gui.addFolder('Ethereal Blur');
        controllers.blurIntensity = blurFolder.add(compositeMaterial.uniforms.uIntensity, 'value', 0, 2, 0.01).name('intensity');

        const filmGrainFolder = gui.addFolder('Film Grain');
        filmGrainFolder.add(compositeMaterial.uniforms.uNoiseAlpha, 'value', 0, 0.2, 0.001).name('intensity');
        
        const fluidFolder = gui.addFolder('Cursor Ripples');
        fluidFolder.add(uniforms.uRippleStrength, 'value', 0, 0.5, 0.005).name('strength');
        fluidFolder.add(rippleUniforms.uDamping, 'value', 0.8, 0.999, 0.001).name('damping');
        fluidFolder.add(rippleUniforms.uBrushSize, 'value', 0.01, 0.2, 0.001).name('brush size');
        fluidFolder.add(rippleUniforms.uTrailPersistence, 'value', 0.8, 0.999, 0.001).name('trail');

        const displacementFolder = gui.addFolder('Displacement');
        controllers.noiseType = displacementFolder.add(displacementParams, 'type', ['Perlin', 'Simplex', 'Worley', 'FBM', 'Smooth']).name('algorithm').onChange((value: string) => {
            uniforms.uNoiseType.value = noiseTypes[value as keyof typeof noiseTypes];
        });

        const waveFolder = gui.addFolder('Large Waves');
        controllers.bigWavesElevation = waveFolder.add(uniforms.uBigWavesElevation, 'value', 0, 0.5, 0.001).name('elevation');
        controllers.bigWavesFrequencyX = waveFolder.add(uniforms.uBigWavesFrequency.value, 'x', 0, 5, 0.01).name('frequencyX');
        controllers.bigWavesFrequencyY = waveFolder.add(uniforms.uBigWavesFrequency.value, 'y', 0, 5, 0.01).name('frequencyY');
        controllers.bigWavesSpeed = waveFolder.add(uniforms.uBigWavesSpeed, 'value', 0, 1, 0.005).name('speed');
        
        const smallWaveFolder = gui.addFolder('Fine Ripples');
        controllers.smallWavesElevation = smallWaveFolder.add(uniforms.uSmallWavesElevation, 'value', 0, 0.3, 0.001).name('elevation');
        controllers.smallWavesFrequency = smallWaveFolder.add(uniforms.uSmallWavesFrequency, 'value', 0, 20, 0.01).name('frequency');
        controllers.smallWavesSpeed = smallWaveFolder.add(uniforms.uSmallWavesSpeed, 'value', 0, 1, 0.005).name('speed');
        controllers.smallWavesIterations = smallWaveFolder.add(uniforms.uSmallWavesIterations, 'value', 1, 8, 1).name('iterations');

        const colorFolder = gui.addFolder('Colors');
        controllers.depthColor = colorFolder.addColor(colorParams, 'depthColor').name('depth').onChange((value: string) => {
            uniforms.uDepthColor.value.set(value);
        });
        controllers.surfaceColor = colorFolder.addColor(colorParams, 'surfaceColor').name('surface').onChange((value: string) => {
            uniforms.uSurfaceColor.value.set(value);
        });
        controllers.colorOffset = colorFolder.add(uniforms.uColorOffset, 'value', 0, 1, 0.001).name('offset');
        controllers.colorMultiplier = colorFolder.add(uniforms.uColorMultiplier, 'value', 0, 10, 0.01).name('multiplier');

        blurFolder.close();
        filmGrainFolder.close();
        fluidFolder.close();
        displacementFolder.close();
        waveFolder.close();
        smallWaveFolder.close();
        colorFolder.close();

        applyPreset(presetsRef.current["Serene Twilight"]);

        const fluidMousePosition = new THREE.Vector2(0.5, 0.5);
        const handleMouseMove = (event: MouseEvent) => {
            fluidMousePosition.x = event.clientX / sizes.width;
            fluidMousePosition.y = 1.0 - (event.clientY / sizes.height);
        };
        const handleResize = () => {
            sizes.width = currentMount.clientWidth;
            sizes.height = currentMount.clientHeight;
            camera.aspect = sizes.width / sizes.height;
            camera.updateProjectionMatrix();
            renderer.setSize(sizes.width, sizes.height);
            renderer.setPixelRatio(qualitySettings[perfParams.quality].pixelRatio);
            uniforms.uCameraAspect.value = camera.aspect;

            // Update post-processing FBOs
            const blurDownsample = settings.blurDownsample;
            sceneFBO.setSize(sizes.width, sizes.height);
            blurFBO1.setSize(sizes.width / blurDownsample, sizes.height / blurDownsample);
            blurFBO2.setSize(sizes.width / blurDownsample, sizes.height / blurDownsample);
            blurHorizontalMaterial.uniforms.uDirection.value.set(1 / (sizes.width / blurDownsample), 0);
            blurVerticalMaterial.uniforms.uDirection.value.set(0, 1 / (sizes.height / blurDownsample));
        };
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('resize', handleResize);

        const clock = new THREE.Clock();
        let animationFrameId: number;
        const tick = () => {
            const elapsedTime = clock.getElapsedTime();
            uniforms.uTime.value = elapsedTime;
            compositeMaterial.uniforms.uTime.value = elapsedTime;
            
            // Update fluid sim mouse
            rippleUniforms.uMouse.value.copy(fluidMousePosition);

            // Ping-pong rendering for fluid simulation
            renderer.setRenderTarget(fbo2);
            rippleUniforms.uPrevTexture.value = fbo1.texture;
            renderer.render(fluidScene, fluidCamera);
            
            const temp = fbo1;
            fbo1 = fbo2;
            fbo2 = temp;

            // --- Post-processing ---
            // 1. Render main scene to texture
            renderer.setRenderTarget(sceneFBO);
            uniforms.uDisplacementMap.value = fbo1.texture;
            renderer.render(scene, camera);

            // 2. Horizontal Blur Pass
            postMesh.material = blurHorizontalMaterial;
            blurHorizontalMaterial.uniforms.uTexture.value = sceneFBO.texture;
            renderer.setRenderTarget(blurFBO1);
            renderer.render(postScene, postCamera);

            // 3. Vertical Blur Pass
            postMesh.material = blurVerticalMaterial;
            blurVerticalMaterial.uniforms.uTexture.value = blurFBO1.texture;
            renderer.setRenderTarget(blurFBO2);
            renderer.render(postScene, postCamera);

            // 4. Composite to screen
            postMesh.material = compositeMaterial;
            compositeMaterial.uniforms.uSceneTexture.value = sceneFBO.texture;
            compositeMaterial.uniforms.uBlurTexture.value = blurFBO2.texture;
            renderer.setRenderTarget(null);
            renderer.render(postScene, postCamera);
            
            animationFrameId = window.requestAnimationFrame(tick);
        };
        tick();
        
        return () => {
            window.removeEventListener('resize', handleResize);
            window.removeEventListener('mousemove', handleMouseMove);
            window.cancelAnimationFrame(animationFrameId);
            renderer.dispose();
            geometry.dispose();
            material.dispose();
            fluidGeometry.dispose();
            rippleMaterial.dispose();
            fbo1.dispose();
            fbo2.dispose();
            // Dispose post-processing resources
            sceneFBO.dispose();
            blurFBO1.dispose();
            blurFBO2.dispose();
            postGeometry.dispose();
            blurHorizontalMaterial.dispose();
            blurVerticalMaterial.dispose();
            compositeMaterial.dispose();
            gui.destroy();
            if (currentMount && renderer.domElement) {
                currentMount.removeChild(renderer.domElement);
            }
        };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return <div ref={mountRef} className="absolute top-0 left-0 w-full h-full" />;
};

export default ShaderCanvas;
