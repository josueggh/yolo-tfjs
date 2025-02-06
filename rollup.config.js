import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import { terser } from 'rollup-plugin-terser';

export default [
  // UMD build for browsers / CDN usage.
  {
    input: 'src/index.ts',
    output: {
      file: 'dist/yolo.umd.js',
      format: 'umd',
      name: 'YOLO', // This is the global variable name.
      exports: 'default',
      sourcemap: true,
      globals: {
        '@tensorflow/tfjs': 'tf',
        '@tensorflow/tfjs-backend-webgl': 'tf'
      }
    },
    external: [
      '@tensorflow/tfjs',
      '@tensorflow/tfjs-backend-webgl'
    ],
    plugins: [
      resolve(),
      commonjs(),
      typescript({tsconfig: './tsconfig.json'}),
      terser(), // Optional: minify the bundle.
    ],
  },
  // ESM build for bundlers.
  {
    input: 'src/index.ts',
    output: {
      file: 'dist/yolo.esm.js',
      format: 'esm',
      sourcemap: true,
    },
    external: [
      '@tensorflow/tfjs',
      '@tensorflow/tfjs-backend-webgl'
    ],
    plugins: [
      resolve(),
      commonjs(),
      typescript({tsconfig: './tsconfig.json'}),
      terser(),
    ],
  },
];
