// WC09 — surgical lint config for kolm.ai
// Scope: src/, cli/, scripts/, tests/, workers/ — Node + browser JS/CJS/MJS.
// Style rules are intentionally OFF; only correctness/bug rules at error level.
// Ignored: public/ (browser HTML+JS, separate concerns), data/, coverage/,
// apps/trainer/ (Python), packages/vscode-kolm-rag/ (TypeScript, separate config),
// node_modules/, md-links-test/, sdk/python/, sdk/rust/, sdk/c/.

import globals from 'globals';

export default [
  {
    ignores: [
      'public/**',
      'node_modules/**',
      'md-links-test/**',
      'data/**',
      'coverage/**',
      'apps/trainer/**',
      'packages/vscode-kolm-rag/**',
      'sdk/python/**',
      'sdk/rust/**',
      'sdk/c/**',
      'sdk/**/node_modules/**',
      '**/node_modules/**',
      '**/__pycache__/**',
      '**/*.min.js',
      '**/dist/**',
      '**/build/**',
    ],
  },
  {
    files: [
      'src/**/*.js',
      'cli/**/*.js',
      'cli/**/*.mjs',
      'tests/**/*.js',
      'workers/**/*.js',
      'workers/**/*.mjs',
    ],
    languageOptions: {
      ecmaVersion: 2024,
      sourceType: 'module',
      globals: {
        ...globals.node,
        ...globals.browser,
        ...globals.es2024,
      },
    },
    rules: {
      'no-undef': 'error',
      'no-unused-vars': [
        'error',
        { args: 'after-used', argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
      eqeqeq: ['error', 'always', { null: 'ignore' }],
      'no-var': 'error',
      'no-unreachable': 'error',
      'no-dupe-keys': 'error',
      'no-dupe-args': 'error',
      'no-cond-assign': 'error',
      'no-constant-condition': ['error', { checkLoops: false }],
    },
  },
  {
    // CommonJS scripts — scripts/**/*.cjs uses require/module.exports.
    // Browser globals included because several scripts embed Playwright
    // `page.evaluate(() => ...)` callbacks that run in browser context.
    files: ['scripts/**/*.cjs'],
    languageOptions: {
      ecmaVersion: 2024,
      sourceType: 'commonjs',
      globals: {
        ...globals.node,
        ...globals.browser,
        ...globals.es2024,
      },
    },
    rules: {
      'no-undef': 'error',
      'no-unused-vars': [
        'error',
        { args: 'after-used', argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
      eqeqeq: ['error', 'always', { null: 'ignore' }],
      'no-var': 'error',
      'no-unreachable': 'error',
      'no-dupe-keys': 'error',
      'no-dupe-args': 'error',
      'no-cond-assign': 'error',
      'no-constant-condition': ['error', { checkLoops: false }],
    },
  },
  {
    // scripts/**/*.{js,mjs} — ESM scripts (e.g. .mjs files).
    // Browser globals included for the same Playwright reason.
    files: ['scripts/**/*.js', 'scripts/**/*.mjs'],
    languageOptions: {
      ecmaVersion: 2024,
      sourceType: 'module',
      globals: {
        ...globals.node,
        ...globals.browser,
        ...globals.es2024,
      },
    },
    rules: {
      'no-undef': 'error',
      'no-unused-vars': [
        'error',
        { args: 'after-used', argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
      eqeqeq: ['error', 'always', { null: 'ignore' }],
      'no-var': 'error',
      'no-unreachable': 'error',
      'no-dupe-keys': 'error',
      'no-dupe-args': 'error',
      'no-cond-assign': 'error',
      'no-constant-condition': ['error', { checkLoops: false }],
    },
  },
];
