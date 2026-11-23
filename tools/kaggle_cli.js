/**
 * Turnkey Kaggle CLI for Windows environments.
 * Uses official Kaggle REST API with OAuth Access Token (~/.kaggle/access_token).
 */

const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');

// 1. Resolve Access Token
function getAccessToken() {
  const tokenPath = path.join(process.env.USERPROFILE || 'C:/Users/PC', '.kaggle', 'access_token');
  if (process.env.KAGGLE_API_TOKEN) {
    return process.env.KAGGLE_API_TOKEN.trim();
  }
  if (fs.existsSync(tokenPath)) {
    return fs.readFileSync(tokenPath, 'utf8').trim();
  }
  throw new Error(`Kaggle access token not found at: ${tokenPath}`);
}

// 2. Generic Kaggle API Caller
function callKaggleApi(endpoint, body = null, method = 'POST') {
  const token = getAccessToken();
  return new Promise((resolve, reject) => {
    const isV1Service = endpoint.includes('.');
    const reqPath = isV1Service ? `/v1/${endpoint}` : endpoint;
    const reqData = body ? JSON.stringify(body) : null;

    const headers = {
      'Authorization': 'Bearer ' + token,
      'User-Agent': 'kaggle-api/v1.7.0'
    };
    if (reqData) {
      headers['Content-Type'] = 'application/json';
      headers['Content-Length'] = Buffer.byteLength(reqData);
    }

    const req = https.request({
      hostname: 'api.kaggle.com',
      path: reqPath,
      method: method,
      headers: headers
    }, (res) => {
      let respBody = '';
      res.on('data', chunk => respBody += chunk);
      res.on('end', () => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          try {
            resolve(respBody ? JSON.parse(respBody) : {});
          } catch (e) {
            resolve(respBody);
          }
        } else {
          reject(new Error(`Kaggle API Error (${res.statusCode} ${res.statusMessage}): ${respBody}`));
        }
      });
    });

    req.on('error', reject);
    if (reqData) req.write(reqData);
    req.end();
  });
}

// 3. User Introspection
async function getUsername() {
  const token = getAccessToken();
  const resp = await callKaggleApi('security.OAuthService/IntrospectToken', { token });
  if (!resp.active || !resp.username) {
    throw new Error('Access token is inactive or invalid: ' + JSON.stringify(resp));
  }
  return resp.username;
}

// 4. Resumable GCS File Uploader with Progress
function uploadFileToGcs(createUrl, filePath) {
  return new Promise((resolve, reject) => {
    const stat = fs.statSync(filePath);
    const totalBytes = stat.size;
    const parsed = new URL(createUrl);
    const fileName = path.basename(filePath);

    let uploadedBytes = 0;
    let lastLogTime = Date.now();
    let lastLoggedBytes = 0;
    const startTime = Date.now();

    const fileStream = fs.createReadStream(filePath, { highWaterMark: 4 * 1024 * 1024 });

    const req = https.request({
      hostname: parsed.hostname,
      path: parsed.pathname + parsed.search,
      method: 'PUT',
      headers: {
        'Content-Length': totalBytes,
        'Content-Type': 'application/octet-stream'
      }
    }, (res) => {
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => {
        if (res.statusCode === 200 || res.statusCode === 201) {
          const totalSec = (Date.now() - startTime) / 1000;
          const avgMBps = (totalBytes / (1024 * 1024)) / Math.max(0.1, totalSec);
          console.log(`\n  [OK] Upload complete: ${fileName} (${(totalBytes / (1024*1024)).toFixed(1)} MB in ${totalSec.toFixed(1)}s, avg ${avgMBps.toFixed(2)} MB/s)`);
          resolve(body);
        } else {
          reject(new Error(`GCS Upload Failed (${res.statusCode}): ${body}`));
        }
      });
    });

    req.on('error', reject);

    fileStream.on('data', chunk => {
      uploadedBytes += chunk.length;
      const now = Date.now();
      if (now - lastLogTime >= 2000 || uploadedBytes === totalBytes) {
        const pct = ((uploadedBytes / totalBytes) * 100).toFixed(1);
        const chunkSec = (now - lastLogTime) / 1000;
        const currentMBps = ((uploadedBytes - lastLoggedBytes) / (1024 * 1024)) / Math.max(0.1, chunkSec);
        const totalMb = (uploadedBytes / (1024 * 1024)).toFixed(1);
        const maxMb = (totalBytes / (1024 * 1024)).toFixed(1);
        process.stdout.write(`\r  -> ${fileName}: ${pct}% [${totalMb}/${maxMb} MB] @ ${currentMBps.toFixed(2)} MB/s `);
        lastLogTime = now;
        lastLoggedBytes = uploadedBytes;
      }
    });

    fileStream.pipe(req);
  });
}

// 5. Upload Single File Blob
async function uploadBlob(filePath) {
  const fileName = path.basename(filePath);
  const stat = fs.statSync(filePath);
  console.log(`\n[Blob] Reserving blob upload for '${fileName}' (${(stat.size / (1024*1024)).toFixed(2)} MB)...`);

  const startResp = await callKaggleApi('blobs.BlobApiService/StartBlobUpload', {
    type: 'DATASET',
    name: fileName,
    contentLength: stat.size,
    lastModifiedEpochSeconds: Math.floor(stat.mtimeMs / 1000)
  });

  if (!startResp.token || !startResp.createUrl) {
    throw new Error('StartBlobUpload did not return token and createUrl: ' + JSON.stringify(startResp));
  }

  await uploadFileToGcs(startResp.createUrl, filePath);
  return startResp.token;
}

// 6. Dataset Creation Command
async function uploadDataset(folderPath, slug = 'stashbench', title = 'STASHBench Dataset') {
  const username = await getUsername();
  console.log('======================================================================');
  console.log(`Kaggle Dataset Upload: ${username}/${slug}`);
  console.log(`Source Folder : ${folderPath}`);
  console.log(`Dataset Title : ${title}`);
  console.log('======================================================================');

  // Discover candidate files in folderPath (skip pool dir itself, since pool.zip is uploaded)
  const entries = fs.readdirSync(folderPath);
  const filesToUpload = [];

  for (const entry of entries) {
    const full = path.join(folderPath, entry);
    const st = fs.statSync(full);
    if (st.isFile()) {
      if (!entry.endsWith('.tmp') && !entry.startsWith('.')) {
        filesToUpload.push(full);
      }
    }
  }

  console.log(`Found ${filesToUpload.length} file(s) to upload:`);
  filesToUpload.forEach(f => console.log('  •', path.basename(f), `(${(fs.statSync(f).size / (1024*1024)).toFixed(2)} MB)`));

  const uploadedFileRefs = [];
  for (const f of filesToUpload) {
    const token = await uploadBlob(f);
    uploadedFileRefs.push({ token: token });
  }

  console.log('\n[Dataset] Finalizing dataset creation on Kaggle backend...');
  const createReq = {
    title: title,
    slug: slug,
    ownerSlug: username,
    licenseName: 'CC0-1.0',
    isPrivate: true,
    files: uploadedFileRefs
  };

  const createResp = await callKaggleApi('datasets.DatasetApiService/CreateDataset', createReq);
  console.log('\n======================================================================');
  console.log('SUCCESS: Dataset created on Kaggle!');
  console.log(`Dataset URL: https://www.kaggle.com/datasets/${username}/${slug}`);
  console.log('Response   :', JSON.stringify(createResp));
  console.log('======================================================================');
  return createResp;
}

// 7. Kernel Push Command
async function pushKernel(notebookPath, slug = 'stash-search-visualization', title = 'STASH-Search Visualization', datasetSources = []) {
  const username = await getUsername();
  console.log('======================================================================');
  console.log(`Kaggle Kernel Push: ${username}/${slug}`);
  console.log(`Notebook File : ${notebookPath}`);
  console.log(`Accelerator   : GPU (2x Tesla T4)`);
  console.log(`Datasets      : ${datasetSources.join(', ') || 'None'}`);
  console.log('======================================================================');

  const nbContent = fs.readFileSync(notebookPath, 'utf8');

  const formattedDatasets = datasetSources.map(d => d.includes('/') ? d : `${username}/${d}`);
  const kernelReq = {
    slug: `${username}/${slug}`,
    newTitle: title,
    text: nbContent,
    language: 'python',
    kernelType: 'notebook',
    isPrivate: true,
    enableGpu: true,
    machineShape: 'NvidiaTeslaT4',
    enableInternet: true,
    datasetDataSources: formattedDatasets,
    dataset_data_sources: formattedDatasets,
    kernelExecutionType: 'SAVE_AND_RUN_ALL',
    kernel_execution_type: 1
  };

  const saveResp = await callKaggleApi('kernels.KernelsApiService/SaveKernel', kernelReq);
  console.log('\n======================================================================');
  console.log('SUCCESS: Kernel successfully pushed and queued on Kaggle!');
  console.log(`Kernel URL: https://www.kaggle.com/code/${username}/${slug}`);
  console.log('Response  :', JSON.stringify(saveResp));
  console.log('======================================================================');
  return saveResp;
}

// 8. Check Status
async function checkStatus(kernelSlug = 'stash-search-visualization') {
  const username = await getUsername();
  const fullSlug = kernelSlug.includes('/') ? kernelSlug : `${username}/${kernelSlug}`;
  const parts = fullSlug.split('/');
  
  const statusResp = await callKaggleApi('kernels.KernelsApiService/GetKernelSessionStatus', {
    user_name: parts[0],
    kernel_slug: parts[1]
  });
  const status = statusResp.status || 'QUEUED';
  console.log('Kernel Status for', fullSlug, ':', { status });

  try {
    const outResp = await callKaggleApi('kernels.KernelsApiService/ListKernelSessionOutput', {
      user_name: parts[0],
      kernel_slug: parts[1]
    });
    if (outResp && outResp.log) {
      try {
        const events = JSON.parse(outResp.log);
        const text = events.map(e => e.data || '').join('');
        console.log('\n--- Recent Logs (Last 1200 chars) ---');
        console.log(text.slice(-1200));
        console.log('-------------------------------------\n');
      } catch (e) {
        console.log('\n--- Raw Log ---');
        console.log(outResp.log.slice(-1200));
      }
    }
  } catch (e) {}

  return statusResp;
}

// 9. CLI Argument Parser
async function main() {
  const args = process.argv.slice(2);
  const command = args[0] || 'help';

  try {
    if (command === 'quota') {
      const username = await getUsername();
      console.log(`Authenticated as: ${username}`);
      const quotaResp = await callKaggleApi('kernels.KernelsApiService/GetAcceleratorQuotaStatistics', { userName: username });
      console.log('Accelerator Quota:', JSON.stringify(quotaResp, null, 2));
    } else if (command === 'upload-dataset') {
      const folder = args[1] || 'D:/MachineLearning/Datasets/STASHBench';
      const slug = args[2] || 'stashbench';
      const title = args[3] || 'STASHBench Dataset';
      await uploadDataset(folder, slug, title);
    } else if (command === 'push-kernel') {
      const nb = args[1] || 'minimal/notebooks/run_stash_visualization_kaggle.ipynb';
      const slug = args[2] || 'stash-search-visualization';
      const title = args[3] || 'STASH-Search Visualization';
      const ds = args[4] ? args[4].split(',') : ['thnhtt/stashbench'];
      await pushKernel(nb, slug, title, ds);
    } else if (command === 'status') {
      const slug = args[1] || 'stash-search-visualization';
      await checkStatus(slug);
    } else {
      console.log('Usage:');
      console.log('  node tools/kaggle_cli.js quota');
      console.log('  node tools/kaggle_cli.js upload-dataset [folder] [slug] [title]');
      console.log('  node tools/kaggle_cli.js push-kernel [notebook] [slug] [title] [datasets]');
      console.log('  node tools/kaggle_cli.js status [kernel-slug]');
    }
  } catch (err) {
    console.error('\n[Error]', err.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}
