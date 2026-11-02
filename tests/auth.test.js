import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { rmSyncBestEffort } from './_spawn-helpers.js';

test('admin fallback key is disabled in production-like hosts', async (t) => {
  const savedEnv = {
    ADMIN_KEY: process.env.ADMIN_KEY,
    NODE_ENV: process.env.NODE_ENV,
    AWS_EXECUTION_ENV: process.env.AWS_EXECUTION_ENV,
    VERCEL: process.env.VERCEL,
    AWS_LAMBDA_FUNCTION_NAME: process.env.AWS_LAMBDA_FUNCTION_NAME,
    KOLM_DATA_DIR: process.env.KOLM_DATA_DIR,
    KOLM_ARTIFACT_DIR: process.env.KOLM_ARTIFACT_DIR,
    KOLM_STORE_DRIVER: process.env.KOLM_STORE_DRIVER,
    KOLM_DB_PATH: process.env.KOLM_DB_PATH,
    KOLM_ALLOW_JSON_STORE: process.env.KOLM_ALLOW_JSON_STORE,
    RECIPE_RECEIPT_SECRET: process.env.RECIPE_RECEIPT_SECRET,
    KOLM_ARTIFACT_SECRET: process.env.KOLM_ARTIFACT_SECRET,
  };
  const testDataDir = path.join(os.tmpdir(), `kolm-auth-${process.pid}-${Date.now()}`);
  const testArtifactDir = path.join(os.tmpdir(), `kolm-artifacts-${process.pid}-${Date.now()}`);
  fs.mkdirSync(testDataDir, { recursive: true });
  fs.mkdirSync(testArtifactDir, { recursive: true });

  process.env.KOLM_DATA_DIR = testDataDir;
  delete process.env.KOLM_ARTIFACT_DIR;
  delete process.env.ADMIN_KEY;
  delete process.env.NODE_ENV;
  delete process.env.AWS_EXECUTION_ENV;
  delete process.env.VERCEL;
  delete process.env.AWS_LAMBDA_FUNCTION_NAME;
  delete process.env.RECIPE_RECEIPT_SECRET;
  delete process.env.KOLM_ARTIFACT_SECRET;
  delete process.env.KOLM_STORE_DRIVER;
  delete process.env.KOLM_DB_PATH;
  delete process.env.KOLM_ALLOW_JSON_STORE;

  t.after(() => {
    for (const [key, value] of Object.entries(savedEnv)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    // Best-effort: in-process sqlite handle from src/store.js can hold
    // testDataDir for a beat after the test ends. Don't fail the test on
    // Windows EPERM — let the OS reap the tmp dir on next reboot.
    rmSyncBestEffort(testDataDir);
    rmSyncBestEffort(testArtifactDir);
  });

  const { adminApiKey, isProductionRuntime } = await import('../src/auth.js');
  const { effectiveReceiptSecret, DEV_RECEIPT_SECRET, runtimeReadiness } = await import('../src/env.js');

  assert.equal(isProductionRuntime(), false);
  // Wave 99 P0-3: ks_admin_change_me hardcoded fallback removed.
  // adminApiKey() now returns null when ADMIN_KEY env is unset, even in dev.
  assert.equal(adminApiKey(), null);
  assert.equal(effectiveReceiptSecret(), DEV_RECEIPT_SECRET);
  assert.equal(runtimeReadiness().status, 'ready');

  process.env.AWS_EXECUTION_ENV = 'AWS_ECS_FARGATE';
  assert.equal(isProductionRuntime(), true);
  assert.equal(adminApiKey(), null);
  assert.equal(effectiveReceiptSecret(), null);
  assert.equal(runtimeReadiness().status, 'not_ready');

  delete process.env.AWS_EXECUTION_ENV;
  process.env.VERCEL = '1';
  assert.equal(isProductionRuntime(), true);
  assert.equal(adminApiKey(), null);
  assert.equal(effectiveReceiptSecret(), null);

  delete process.env.VERCEL;
  process.env.AWS_LAMBDA_FUNCTION_NAME = 'kolm-api';
  assert.equal(isProductionRuntime(), true);
  assert.equal(adminApiKey(), null);
  assert.equal(effectiveReceiptSecret(), null);

  process.env.ADMIN_KEY = 'ks_live_admin';
  process.env.RECIPE_RECEIPT_SECRET = 'ks_receipt_live';
  process.env.KOLM_ARTIFACT_DIR = testArtifactDir;
  assert.equal(adminApiKey(), 'ks_live_admin');
  assert.equal(effectiveReceiptSecret(), null);
  assert.equal(runtimeReadiness().status, 'not_ready');

  process.env.RECIPE_RECEIPT_SECRET = 'ks_receipt_' + 'a'.repeat(48);
  process.env.KOLM_STORE_DRIVER = 'sqlite';
  process.env.KOLM_DB_PATH = path.join(testDataDir, 'store.sqlite');
  assert.equal(effectiveReceiptSecret(), process.env.RECIPE_RECEIPT_SECRET);
  assert.equal(runtimeReadiness().status, 'ready');

  // resolveArtifactDir() auto-bootstraps via ensureDirSync — a missing path is
  // created on-the-fly so artifact_dir readiness stays 'ready'. The fail-closed
  // case would require an explicitly unwritable parent, which is OS-specific.
  process.env.KOLM_ARTIFACT_DIR = path.join(testArtifactDir, 'auto-created');
  assert.equal(runtimeReadiness().status, 'ready');
});
