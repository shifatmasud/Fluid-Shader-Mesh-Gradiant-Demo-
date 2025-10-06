import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
import GUI from 'lil-gui';
import type { ShaderUniforms } from '../types';

// GLSL Shaders
const vertexShader = `
  uniform float uTime;
  uniform vec2 uMouse;
  uniform float uCameraAspect;
  uniform int uNoiseType; // 0: Perlin, 1: Simplex, 2: Worley

  uniform float uBigWavesElevation;
  uniform vec2 uBigWavesFrequency;
  uniform float uBigWavesSpeed;
  uniform float uSmallWavesElevation;
  uniform float uSmallWavesFrequency;
  uniform float uSmallWavesSpeed;
  uniform float uSmallWavesIterations;

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
    vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
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
  
  float noise(vec3 p) {
    if (uNoiseType == 0) return cnoise(p);
    if (uNoiseType == 1) return snoise(p);
    if (uNoiseType == 2) return 1.0 - worley(p) * 2.0;
    return 0.0;
  }

  varying float vElevation;

  void main() {
    vec4 modelPosition = modelMatrix * vec4(position, 1.0);

    // Big waves
    float elevation = noise(vec3(
        modelPosition.x * uBigWavesFrequency.x + uTime * uBigWavesSpeed,
        modelPosition.y * uBigWavesFrequency.y + uTime * uBigWavesSpeed,
        uTime * 0.1
    )) * uBigWavesElevation;

    // Small waves (FBM)
    for(float i = 1.0; i <= uSmallWavesIterations; i++) {
        elevation += noise(vec3(
            modelPosition.x * uSmallWavesFrequency * i + uTime * uSmallWavesSpeed,
            modelPosition.y * uSmallWavesFrequency * i + uTime * uSmallWavesSpeed,
            uTime * 0.2
        )) * (uSmallWavesElevation / i);
    }
    
    // Mouse interaction
    float mouseDistance = distance(vec2(uMouse.x * (2.0 / uCameraAspect), uMouse.y * 2.0), modelPosition.xy);
    float mouseEffect = 1.0 - smoothstep(0.0, 0.4, mouseDistance);
    elevation += mouseEffect * 0.15;

    modelPosition.z = elevation;
    vElevation = elevation;

    gl_Position = projectionMatrix * viewMatrix * modelPosition;
  }
`;

const fragmentShader = `
  uniform vec3 uDepthColor;
  uniform vec3 uSurfaceColor;
  uniform float uColorOffset;
  uniform float uColorMultiplier;

  varying float vElevation;

  void main() {
    float mixStrength = (vElevation + uColorOffset) * uColorMultiplier;
    vec3 color = mix(uDepthColor, uSurfaceColor, mixStrength);
    gl_FragColor = vec4(color, 1.0);
  }
`;

const hexToRgb = (hex: string) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16) / 255,
        g: parseInt(result[2], 16) / 255,
        b: parseInt(result[3], 16) / 255
    } : { r: 0, g: 0, b: 0 };
};

let presets = {
  "Cool Blue": {
    noiseType: 1, // Simplex
    uBigWavesElevation: 0.05, uBigWavesFrequency: { x: 0.6, y: 0.4 }, uBigWavesSpeed: 0.03,
    uSmallWavesElevation: 0.04, uSmallWavesFrequency: 2.5, uSmallWavesSpeed: 0.1, uSmallWavesIterations: 4.0,
    depthColor: '#0A2463', surfaceColor: '#ADD8E6', uColorOffset: 0.1, uColorMultiplier: 3.5,
  },
  "Mystery Purple": {
    noiseType: 0, // Perlin
    uBigWavesElevation: 0.1, uBigWavesFrequency: { x: 0.2, y: 0.1 }, uBigWavesSpeed: 0.01,
    uSmallWavesElevation: 0.02, uSmallWavesFrequency: 1.2, uSmallWavesSpeed: 0.04, uSmallWavesIterations: 5.0,
    depthColor: '#240046', surfaceColor: '#C77DFF', uColorOffset: 0.25, uColorMultiplier: 4.0,
  },
  "Cherry Pink": {
    noiseType: 1, // Simplex
    uBigWavesElevation: 0.03, uBigWavesFrequency: { x: 1.0, y: 0.8 }, uBigWavesSpeed: 0.06,
    uSmallWavesElevation: 0.05, uSmallWavesFrequency: 5.0, uSmallWavesSpeed: 0.2, uSmallWavesIterations: 4.0,
    depthColor: '#D90368', surfaceColor: '#FFDDE2', uColorOffset: 0.05, uColorMultiplier: 3.0,
  },
  "Soft Sunbeam": {
    noiseType: 0, // Perlin
    uBigWavesElevation: 0.02, uBigWavesFrequency: { x: 0.5, y: 0.5 }, uBigWavesSpeed: 0.025,
    uSmallWavesElevation: 0.04, uSmallWavesFrequency: 4.0, uSmallWavesSpeed: 0.15, uSmallWavesIterations: 3.0,
    depthColor: '#FFC947', surfaceColor: '#FEF9E7', uColorOffset: 0.15, uColorMultiplier: 2.5,
  },
  "Lush Grassland": {
    noiseType: 1, // Simplex
    uBigWavesElevation: 0.06, uBigWavesFrequency: { x: 0.4, y: 0.2 }, uBigWavesSpeed: 0.02,
    uSmallWavesElevation: 0.03, uSmallWavesFrequency: 3.0, uSmallWavesSpeed: 0.07, uSmallWavesIterations: 4.0,
    depthColor: '#004B23', surfaceColor: '#C1FF72', uColorOffset: 0.2, uColorMultiplier: 3.0,
  },
};
type Preset = typeof presets[keyof typeof presets];
type Presets = typeof presets;

const ShaderCanvas: React.FC = () => {
    const mountRef = useRef<HTMLDivElement>(null);
    
    useEffect(() => {
        if (!mountRef.current) return;

        const currentMount = mountRef.current;
        const scene = new THREE.Scene();
        const sizes = { width: currentMount.clientWidth, height: currentMount.clientHeight };
        const camera = new THREE.PerspectiveCamera(75, sizes.width / sizes.height, 0.1, 100);
        camera.position.set(0, 0, 1.8);
        scene.add(camera);
        
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(sizes.width, sizes.height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        currentMount.appendChild(renderer.domElement);
        
        const waterColors = { depthColor: '#d1e0ff', surfaceColor: '#fff4e0' };
        const displacementParams = { type: 'Simplex' };
        const noiseTypes = { Perlin: 0, Simplex: 1, Worley: 2 };
        
        const uniforms: ShaderUniforms = {
            uTime: { value: 0 },
            uMouse: { value: { x: 9999, y: 9999 } },
            uCameraAspect: { value: camera.aspect },
            uNoiseType: { value: noiseTypes.Simplex },
            uBigWavesElevation: { value: 0.08 }, uBigWavesFrequency: { value: { x: 0.6, y: 0.4 } }, uBigWavesSpeed: { value: 0.04 },
            uSmallWavesElevation: { value: 0.06 }, uSmallWavesFrequency: { value: 2.0 }, uSmallWavesSpeed: { value: 0.08 }, uSmallWavesIterations: { value: 3.0 },
            uDepthColor: { value: hexToRgb(waterColors.depthColor) },
            uSurfaceColor: { value: hexToRgb(waterColors.surfaceColor) },
            uColorOffset: { value: 0.3 },
            uColorMultiplier: { value: 2.5 }
        };

        const material = new THREE.ShaderMaterial({ vertexShader, fragmentShader, uniforms });
        const geometry = new THREE.PlaneGeometry(4.5, 4.5, 512, 512);
        const mesh = new THREE.Mesh(geometry, material);
        mesh.rotation.x = -Math.PI * 0.35;
        scene.add(mesh);
        
        const gui = new GUI();
        
        const applyPreset = (preset: Preset) => {
            uniforms.uNoiseType.value = preset.noiseType;
            displacementParams.type = Object.keys(noiseTypes).find(key => noiseTypes[key as keyof typeof noiseTypes] === preset.noiseType) || 'Simplex';
            uniforms.uBigWavesElevation.value = preset.uBigWavesElevation;
            uniforms.uBigWavesFrequency.value.x = preset.uBigWavesFrequency.x;
            uniforms.uBigWavesFrequency.value.y = preset.uBigWavesFrequency.y;
            uniforms.uBigWavesSpeed.value = preset.uBigWavesSpeed;
            uniforms.uSmallWavesElevation.value = preset.uSmallWavesElevation;
            uniforms.uSmallWavesFrequency.value = preset.uSmallWavesFrequency;
            uniforms.uSmallWavesSpeed.value = preset.uSmallWavesSpeed;
            uniforms.uSmallWavesIterations.value = preset.uSmallWavesIterations;
            uniforms.uColorOffset.value = preset.uColorOffset;
            uniforms.uColorMultiplier.value = preset.uColorMultiplier;
            waterColors.depthColor = preset.depthColor;
            waterColors.surfaceColor = preset.surfaceColor;
            uniforms.uDepthColor.value = hexToRgb(preset.depthColor);
            uniforms.uSurfaceColor.value = hexToRgb(preset.surfaceColor);
            gui.controllers.forEach((c) => c.updateDisplay());
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
                const json = JSON.stringify(presets, null, 2);
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
                                presets = newPresets;
                                rebuildPresetsFolder(presets);
                                const firstPreset = Object.values(presets)[0] as Preset;
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

        gui.add(ioControls, 'importPresets').name('Import Presets');
        gui.add(ioControls, 'exportPresets').name('Export Presets');
        
        rebuildPresetsFolder(presets);
        applyPreset(presets["Cool Blue"]);
        
        const displacementFolder = gui.addFolder('Displacement');
        displacementFolder.add(displacementParams, 'type', ['Perlin', 'Simplex', 'Worley']).name('algorithm').onChange((value: string) => {
            uniforms.uNoiseType.value = noiseTypes[value as keyof typeof noiseTypes];
        });

        const waveFolder = gui.addFolder('Large Waves');
        waveFolder.add(uniforms.uBigWavesElevation, 'value', 0, 0.5, 0.001).name('elevation');
        waveFolder.add(uniforms.uBigWavesFrequency.value, 'x', 0, 5, 0.01).name('frequencyX');
        waveFolder.add(uniforms.uBigWavesFrequency.value, 'y', 0, 5, 0.01).name('frequencyY');
        waveFolder.add(uniforms.uBigWavesSpeed, 'value', 0, 1, 0.005).name('speed');
        
        const smallWaveFolder = gui.addFolder('Fine Ripples');
        smallWaveFolder.add(uniforms.uSmallWavesElevation, 'value', 0, 0.3, 0.001).name('elevation');
        smallWaveFolder.add(uniforms.uSmallWavesFrequency, 'value', 0, 20, 0.01).name('frequency');
        smallWaveFolder.add(uniforms.uSmallWavesSpeed, 'value', 0, 1, 0.005).name('speed');
        smallWaveFolder.add(uniforms.uSmallWavesIterations, 'value', 1, 8, 1).name('iterations');

        const colorFolder = gui.addFolder('Colors');
        colorFolder.addColor(waterColors, 'depthColor').name('depth').onChange((v: string) => uniforms.uDepthColor.value = hexToRgb(v));
        colorFolder.addColor(waterColors, 'surfaceColor').name('surface').onChange((v: string) => uniforms.uSurfaceColor.value = hexToRgb(v));
        colorFolder.add(uniforms.uColorOffset, 'value', 0, 1, 0.001).name('offset');
        colorFolder.add(uniforms.uColorMultiplier, 'value', 0, 10, 0.01).name('multiplier');

        displacementFolder.close();
        waveFolder.close();
        smallWaveFolder.close();
        colorFolder.close();

        const mousePosition = new THREE.Vector2(9999, 9999);
        const handleMouseMove = (event: MouseEvent) => {
            mousePosition.x = (event.clientX / sizes.width) * 2 - 1;
            mousePosition.y = -(event.clientY / sizes.height) * 2 + 1;
        };
        const handleResize = () => {
            sizes.width = currentMount.clientWidth;
            sizes.height = currentMount.clientHeight;
            camera.aspect = sizes.width / sizes.height;
            camera.updateProjectionMatrix();
            renderer.setSize(sizes.width, sizes.height);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            uniforms.uCameraAspect.value = camera.aspect;
        };
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('resize', handleResize);

        const clock = new THREE.Clock();
        let animationFrameId: number;
        const tick = () => {
            const elapsedTime = clock.getElapsedTime();
            uniforms.uTime.value = elapsedTime;
            uniforms.uMouse.value.x += (mousePosition.x - uniforms.uMouse.value.x) * 0.05;
            uniforms.uMouse.value.y += (mousePosition.y - uniforms.uMouse.value.y) * 0.05;
            renderer.render(scene, camera);
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