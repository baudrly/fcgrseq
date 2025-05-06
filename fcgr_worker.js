/**
 * Web Worker for FCGR Analyzer
 * Performs heavy computations: FCGR generation and metrics calculation.
 */

// --- Calculation Functions (Copied and adapted from previous script.js) ---

// generateFCGR function
function generateFCGR(sequence, k) {
    const dim = 1 << k;
    const mask = dim - 1;
    const fcgrCounts = new Float32Array(dim * dim).fill(0);
    let x = 0, y = 0;
    let validKmerCount = 0;
    let currentKmerLen = 0;
    const baseMap = { 'A': 0, 'T': 1, 'C': 2, 'G': 3 };

    for (let i = 0; i < sequence.length; i++) {
        const base = sequence[i];
        const val = baseMap[base];
        if (val === undefined) { x = y = currentKmerLen = 0; continue; }
        const xb = val & 1;
        const yb = val >> 1;
        x = ((x << 1) | xb) & mask;
        y = ((y << 1) | yb) & mask;
        currentKmerLen++;
        if (currentKmerLen >= k) {
            fcgrCounts[y * dim + x]++;
            validKmerCount++;
        }
    }

    const fcgrMatrix = new Float32Array(dim * dim).fill(0);
    if (validKmerCount > 0) {
        for (let i = 0; i < fcgrCounts.length; i++) {
            fcgrMatrix[i] = fcgrCounts[i] / validKmerCount;
        }
    }
    // IMPORTANT: Return counts as well for normalization in Haralick/Hu if needed?
    // Or assume normalized matrix is sufficient for JS implementations? Assume normalized for now.
    return { fcgrMatrix, validKmerCount };
}


// quantizeMatrix
function quantizeMatrix(matrix, levels) {
    const quantized = new Uint8Array(matrix.length);
    let minVal = Infinity, maxVal = -Infinity;
    for (let i = 0; i < matrix.length; i++) {
        if (matrix[i] < minVal) minVal = matrix[i];
        if (matrix[i] > maxVal) maxVal = matrix[i];
    }
    const range = maxVal - minVal;
    if (range < 1e-9) return quantized;
    const scale = (levels - 1) / range;
    for (let i = 0; i < matrix.length; i++) {
        quantized[i] = Math.min(levels - 1, Math.round((matrix[i] - minVal) * scale));
    }
    return quantized;
}

// calculateGLCM
function calculateGLCM(quantizedMatrix, dim, levels) {
    const glcm = new Float32Array(levels * levels).fill(0);
    let pairs = 0;
    // Simplified: Distance 1, Angle 0 (Horizontal) ONLY for performance
    for (let y = 0; y < dim; y++) {
        for (let x = 0; x < dim - 1; x++) {
            const level1 = quantizedMatrix[y * dim + x];
            const level2 = quantizedMatrix[y * dim + (x + 1)];
            glcm[level1 * levels + level2]++;
            glcm[level2 * levels + level1]++; // Symmetric
            pairs += 2;
        }
    }
    // Add other angles/distances here if needed, but increases computation significantly
    if (pairs > 0) {
        for (let i = 0; i < glcm.length; i++) glcm[i] /= pairs;
    }
    return glcm;
}

// calculateHaralick
function calculateHaralick(glcm, levels) {
    let contrast = 0, dissimilarity = 0, homogeneity = 0, asm = 0, energy = 0;
    let mean_i = 0, mean_j = 0, std_i = 0, std_j = 0, correlation = 0;
    const px = new Float32Array(levels).fill(0);
    const py = new Float32Array(levels).fill(0);

    for (let i = 0; i < levels; i++) {
        for (let j = 0; j < levels; j++) {
            const p_ij = glcm[i * levels + j];
            px[i] += p_ij; py[j] += p_ij;
        }
    }
    for (let i = 0; i < levels; i++) {
        mean_i += i * px[i]; mean_j += i * py[i]; // Note: Using 'i' for py index
    }
    for (let i = 0; i < levels; i++) {
        std_i += (i - mean_i) ** 2 * px[i];
        std_j += (i - mean_j) ** 2 * py[i]; // Note: Using 'i' for py index
    }
    std_i = Math.sqrt(std_i); std_j = Math.sqrt(std_j);

    for (let i = 0; i < levels; i++) {
        for (let j = 0; j < levels; j++) {
            const p_ij = glcm[i * levels + j];
            if (p_ij > 0) {
                contrast += (i - j) ** 2 * p_ij;
                dissimilarity += Math.abs(i - j) * p_ij;
                homogeneity += p_ij / (1 + (i - j) ** 2);
                asm += p_ij ** 2;
                if (std_i > 1e-7 && std_j > 1e-7) { // Increased epsilon slightly
                     correlation += (i - mean_i) * (j - mean_j) * p_ij / (std_i * std_j);
                }
            }
        }
    }
    energy = Math.sqrt(asm);
    if (std_i < 1e-7 || std_j < 1e-7) correlation = 1.0; // Correlation is 1 for constant

    // Basic check for NaN results before returning
    const results = { contrast, dissimilarity, homogeneity, energy, correlation, asm };
    for(const key in results) {
        if (!isFinite(results[key])) results[key] = 0; // Default to 0 if NaN/Inf
    }
    // Homogeneity, Energy, ASM should be <= 1, Correlation [-1, 1]
    results.homogeneity = Math.min(1, Math.max(0, results.homogeneity));
    results.energy = Math.min(1, Math.max(0, results.energy));
    results.asm = Math.min(1, Math.max(0, results.asm));
    results.correlation = Math.min(1, Math.max(-1, results.correlation));


    return results;
}


// calculateImageMoments
function calculateImageMoments(matrix, dim) {
    const m = { m00: 0, m10: 0, m01: 0, m20: 0, m02: 0, m11: 0, m30: 0, m03: 0, m12: 0, m21: 0 };
     for (let y = 0; y < dim; y++) {
        for (let x = 0; x < dim; x++) {
            const p = matrix[y * dim + x];
            if (p > 1e-9) {
                const x_f = parseFloat(x); // Ensure float arithmetic
                const y_f = parseFloat(y);
                m.m00 += p;
                m.m10 += x_f * p;
                m.m01 += y_f * p;
                m.m20 += x_f * x_f * p;
                m.m02 += y_f * y_f * p;
                m.m11 += x_f * y_f * p;
                m.m30 += x_f * x_f * x_f * p;
                m.m03 += y_f * y_f * y_f * p;
                m.m12 += x_f * y_f * y_f * p;
                m.m21 += x_f * x_f * y_f * p;
            }
        }
    }
    return m;
}


// calculateHuMoments
function calculateHuMoments(matrix, dim) {
    const hu = new Float32Array(7).fill(0);
    const m = calculateImageMoments(matrix, dim);
    const epsilon = 1e-9;

    if (Math.abs(m.m00) < epsilon) return hu;

    const x_bar = m.m10 / m.m00;
    const y_bar = m.m01 / m.m00;

    const mu = { // Central Moments
        mu20: m.m20 - x_bar * m.m10,
        mu02: m.m02 - y_bar * m.m01,
        mu11: m.m11 - y_bar * m.m10, // Or m.m11 - x_bar * m.m01
        mu30: m.m30 - 3 * x_bar * m.m20 + 2 * x_bar * x_bar * m.m10,
        mu03: m.m03 - 3 * y_bar * m.m02 + 2 * y_bar * y_bar * m.m01,
        mu12: m.m12 - 2 * y_bar * m.m11 - x_bar * m.m02 + 2 * y_bar * y_bar * m.m10,
        mu21: m.m21 - 2 * x_bar * m.m11 - y_bar * m.m20 + 2 * x_bar * x_bar * m.m01,
    };

     // Normalized Central Moments (using simplified normalization for JS stability)
     // OpenCV normalization is m.m00 ^ ((p+q)/2 + 1)
     // Let's just use m.m00 for simplicity, as relative values matter more
    const m00_sq = m.m00 * m.m00 + epsilon;
    const m00_2p5 = m.m00 ** 2.5 + epsilon;

    const eta = {
        eta20: mu.mu20 / m00_sq,
        eta02: mu.mu02 / m00_sq,
        eta11: mu.mu11 / m00_sq,
        eta30: mu.mu30 / m00_2p5,
        eta03: mu.mu03 / m00_2p5,
        eta12: mu.mu12 / m00_2p5,
        eta21: mu.mu21 / m00_2p5,
    };


    // Hu Moments calculation
    hu[0] = eta.eta20 + eta.eta02;
    hu[1] = (eta.eta20 - eta.eta02) ** 2 + 4 * eta.eta11 ** 2;
    hu[2] = (eta.eta30 - 3 * eta.eta12) ** 2 + (3 * eta.eta21 - eta.eta03) ** 2;
    hu[3] = (eta.eta30 + eta.eta12) ** 2 + (eta.eta21 + eta.eta03) ** 2;
    hu[4] = (eta.eta30 - 3 * eta.eta12) * (eta.eta30 + eta.eta12) * ((eta.eta30 + eta.eta12) ** 2 - 3 * (eta.eta21 + eta.eta03) ** 2) +
            (3 * eta.eta21 - eta.eta03) * (eta.eta21 + eta.eta03) * (3 * (eta.eta30 + eta.eta12) ** 2 - (eta.eta21 + eta.eta03) ** 2);
    hu[5] = (eta.eta20 - eta.eta02) * ((eta.eta30 + eta.eta12) ** 2 - (eta.eta21 + eta.eta03) ** 2) +
            4 * eta.eta11 * (eta.eta30 + eta.eta12) * (eta.eta21 + eta.eta03);
    hu[6] = (3 * eta.eta21 - eta.eta03) * (eta.eta30 + eta.eta12) * ((eta.eta30 + eta.eta12) ** 2 - 3 * (eta.eta21 + eta.eta03) ** 2) -
            (eta.eta30 - 3 * eta.eta12) * (eta.eta21 + eta.eta03) * (3 * (eta.eta30 + eta.eta12) ** 2 - (eta.eta21 + eta.eta03) ** 2);

    // Log-scale transformation
    const huLog = hu.map(h => {
        const absH = Math.abs(h);
        return absH < 1e-10 ? 0 : -Math.sign(h) * Math.log10(absH); // Adjust epsilon
    });

    // Check for NaN/Infinity in final log results
    for(let i=0; i<huLog.length; i++) {
        if (!isFinite(huLog[i])) huLog[i] = 0;
    }

    return huLog;
}

function calculateFractalDimension(matrix, dim) {
    // Implements a simplified Box Counting algorithm
    //  This is NOT the most efficient.
    const threshold = Math.max(0.01 * matrix.reduce((a, b) => a + b, 0) / matrix.length, 1e-9);
    const binary = matrix.map(v => (v > threshold) ? 1 : 0);  // threshold and create binary array
    let pixels = [];
    for(let y = 0; y < dim; y++) {
        for(let x = 0; x < dim; x++) {
            if (binary[y * dim + x] === 1) {
                pixels.push({ x: x, y: y });
            }
        }
    }

    if(pixels.length === 0) return 0; // Empty image has dimension 0.

    // Scales (powers of 2)
    let scales = [];
    let maxPower = Math.floor(Math.log2(dim)); // Largest scale we can use
    for (let i = maxPower; i > 0; i--) scales.push(Math.pow(2, i));
    // Ensure at least 2 scales for linear regression to function
    if(scales.length < 2) { return 0;}

    // Box counting loop
    let counts = [];
    let validScales = [];
    for(let scale of scales) {
        let nx = Math.ceil(dim / scale);
        let ny = Math.ceil(dim / scale);
        // Basic check to limit insane memory usage on bad data (scale too small)
        if (nx*ny > 1e7) {
            console.warn(`FD: Scale ${scale} resulted in a grid size ${nx}x${ny} too large. Skipping`);
            continue
        }

        let occupied = new Array(nx * ny).fill(0); // Track occupied boxes in flattened form. init with 0!
        let count = 0;

        for(let pixel of pixels) {
            const boxX = Math.floor(pixel.x / scale);
            const boxY = Math.floor(pixel.y / scale);

            if(boxX >= 0 && boxX < nx && boxY >= 0 && boxY < ny && occupied[boxY * nx + boxX] === 0) {
                occupied[boxY * nx + boxX] = 1;  // Set box as occupied!
                count++;
            }
        }
        if(count > 0) {
            counts.push(count);
            validScales.push(scale);
        }
    }
    if(counts.length < 2) { // If fewer than 2 scales, return 0, can't do linear regression.
        return 0;
    }

    // Linear regression to find the slope
    let logCounts = counts.map(c => Math.log(c));
    let negLogScales = validScales.map(s => -Math.log(s));

    // Basic linear regression  (very simplified)
    let nVals = logCounts.length;
    let sumX = negLogScales.reduce((a,b) => a+b, 0); // Sum of NegLogScales
    let sumY = logCounts.reduce((a,b) => a+b, 0);    // Sum of LogCounts
    let sumXY = 0; // Sum of X*Y
    let sumX2 = 0; // Sum of X^2

    for(let i=0; i < nVals; i++) {
        sumXY += negLogScales[i] * logCounts[i];
        sumX2 += negLogScales[i] * negLogScales[i];
    }

    const slope = (nVals * sumXY - sumX * sumY) / (nVals * sumX2 - sumX * sumX); // Direct slope calculation

    // Limited error handling
    if (!isFinite(slope)) return 0; // If slope is NaN or Infinity - return 0
    return slope;
}

// calculateAllMetrics - Now includes Haralick and Hu
function calculateAllMetrics(fcgrMatrix, dim) {
    const n = fcgrMatrix.length;
    const fractalDimension = calculateFractalDimension(fcgrMatrix, dim); // Call it here
    // Metrics dictionary is defined as in the example from before
    const metrics = { mean: 0, variance: 0, skewness: 0, kurtosis: -3, entropy: 0, contrast: 0, dissimilarity: 0, homogeneity: 1, energy: 1, correlation: 1, asm: 1, hu0: 0, hu1: 0, hu2: 0, hu3: 0, hu4: 0, hu5: 0, hu6: 0, fractalDimension: fractalDimension };
 
    const epsilon = 1e-9;

    if (n === 0) return metrics;

    // Basic Stats
    let sum = 0, sumSq = 0, sumCube = 0, sumFourth = 0;
    for (let i = 0; i < n; i++) {
        const p = fcgrMatrix[i];
        if (p > epsilon) {
            sum += p;
            sumSq += p * p;
            sumCube += p * p * p;
            sumFourth += p * p * p * p;
            metrics.entropy -= p * Math.log2(p);
        }
    }
    metrics.mean = sum / n;
    metrics.variance = (sumSq / n) - (metrics.mean * metrics.mean);

    if (metrics.variance > epsilon) {
        const stdDev = Math.sqrt(metrics.variance);
        const centralMoment3 = (sumCube / n) - 3 * metrics.mean * (sumSq / n) + 2 * metrics.mean ** 3;
        metrics.skewness = centralMoment3 / (stdDev ** 3);
        const centralMoment4 = (sumFourth / n) - 4 * metrics.mean * (sumCube / n) + 6 * metrics.mean ** 2 * (sumSq / n) - 3 * metrics.mean ** 4;
        metrics.kurtosis = (centralMoment4 / (metrics.variance ** 2)) - 3;
    }
     // Ensure basic stats are finite
     for(const key of ['mean', 'variance', 'skewness', 'kurtosis', 'entropy']) {
         if(!isFinite(metrics[key])) metrics[key] = (key === 'kurtosis' ? -3 : 0);
     }

    // Haralick Features
    try {
        const levels = 16; // Keep levels low for performance
        const quantized = quantizeMatrix(fcgrMatrix, levels);
        // Check if quantized matrix is constant (all same value)
        let isQuantizedConstant = true;
        if (quantized.length > 0) {
            const firstVal = quantized[0];
            for(let i = 1; i < quantized.length; i++) {
                if (quantized[i] !== firstVal) {
                    isQuantizedConstant = false;
                    break;
                }
            }
        }

        if (!isQuantizedConstant) {
            const glcm = calculateGLCM(quantized, dim, levels);
            const hf = calculateHaralick(glcm, levels);
            Object.assign(metrics, hf); // Assign calculated Haralick features
        } // else: keep default Haralick values (homogeneity=1, energy=1, correlation=1, others=0)

    } catch (e) {
        console.error("Worker: Haralick calculation failed:", e);
        // Keep default Haralick values
    }

    // Hu Moments
    try {
        const huLog = calculateHuMoments(fcgrMatrix, dim);
        for (let i = 0; i < huLog.length; i++) {
            metrics[`hu${i}`] = huLog[i];
        }
    } catch (e) {
         console.error("Worker: Hu moments calculation failed:", e);
         // Keep default Hu values (all 0)
    }

    return metrics;
}


// --- Worker Message Handler ---
self.onmessage = function(e) {
    const { id, sequence, k, dim } = e.data;
    // console.log(`Worker received job for ID: ${id}, k=${k}`);

    let result = { id, k, dim, fcgrMatrix: null, validKmerCount: 0, metrics: null, error: null };
    let transferList = []; // Buffers to transfer back

    try {
        const fcgrResult = generateFCGR(sequence, k);
        result.validKmerCount = fcgrResult.validKmerCount;

        if (result.validKmerCount > 0) {
            result.fcgrMatrix = fcgrResult.fcgrMatrix; // Float32Array
            result.metrics = calculateAllMetrics(result.fcgrMatrix, dim);
            transferList.push(result.fcgrMatrix.buffer); // Mark buffer for transfer
        } else {
            result.error = `No valid ${k}-mers found.`;
            // Still create a zero matrix to send back if needed? Or handle null in main thread?
            // Let's send back null and handle it there.
            result.metrics = calculateAllMetrics(new Float32Array(dim*dim), dim); // Calc metrics on zero matrix
        }

        // Post result back to main thread
        self.postMessage(result, transferList);

    } catch (error) {
        console.error(`Worker error for ID ${id}:`, error);
        result.error = error.message || "Unknown worker error";
        // metrics will be default zeros from calculateAllMetrics if fcgrMatrix is null
        result.metrics = calculateAllMetrics(new Float32Array(dim*dim), dim);
        self.postMessage(result); // Post error back
    }
};

// Log worker start
// console.log("FCGR Worker initialized.");
