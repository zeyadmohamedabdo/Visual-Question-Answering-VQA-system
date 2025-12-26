/**
 * VQA System - Frontend JavaScript
 * Handles image upload, API communication, and result display
 */

// ============================================
// Configuration
// ============================================
const CONFIG = {
    apiBaseUrl: 'http://localhost:8000',
    maxFileSize: 10 * 1024 * 1024, // 10MB
    supportedFormats: ['image/jpeg', 'image/png', 'image/webp', 'image/gif'],
    topK: 5
};

// ============================================
// DOM Elements
// ============================================
const elements = {
    // Upload
    dropZone: document.getElementById('dropZone'),
    imageInput: document.getElementById('imageInput'),
    previewContainer: document.getElementById('previewContainer'),
    imagePreview: document.getElementById('imagePreview'),
    removeImage: document.getElementById('removeImage'),

    // Question
    questionInput: document.getElementById('questionInput'),
    charCount: document.getElementById('charCount'),
    exampleBtns: document.querySelectorAll('.example-btn'),
    submitBtn: document.getElementById('submitBtn'),

    // Status
    statusContainer: document.getElementById('statusContainer'),
    statusMessage: document.getElementById('statusMessage'),

    // Results
    resultsPlaceholder: document.getElementById('resultsPlaceholder'),
    resultsContent: document.getElementById('resultsContent'),
    topAnswer: document.getElementById('topAnswer'),
    confidenceFill: document.getElementById('confidenceFill'),
    confidenceText: document.getElementById('confidenceText'),
    altAnswersList: document.getElementById('altAnswersList'),

    // Loading
    loadingOverlay: document.getElementById('loadingOverlay')
};

// ============================================
// State
// ============================================
let state = {
    selectedFile: null,
    isLoading: false
};

// ============================================
// Utility Functions
// ============================================

/**
 * Show status message
 */
function showStatus(message, type = 'error') {
    elements.statusContainer.style.display = 'block';
    elements.statusMessage.textContent = message;
    elements.statusMessage.className = `status-message ${type}`;

    // Auto-hide after 5 seconds
    setTimeout(() => {
        elements.statusContainer.style.display = 'none';
    }, 5000);
}

/**
 * Hide status message
 */
function hideStatus() {
    elements.statusContainer.style.display = 'none';
}

/**
 * Validate file
 */
function validateFile(file) {
    if (!file) {
        return { valid: false, error: 'No file selected' };
    }

    if (!CONFIG.supportedFormats.includes(file.type)) {
        return {
            valid: false,
            error: `Unsupported format. Please use: ${CONFIG.supportedFormats.map(f => f.split('/')[1]).join(', ')}`
        };
    }

    if (file.size > CONFIG.maxFileSize) {
        return {
            valid: false,
            error: `File too large. Maximum size is ${CONFIG.maxFileSize / (1024 * 1024)}MB`
        };
    }

    return { valid: true };
}

/**
 * Update submit button state
 */
function updateSubmitButton() {
    const hasImage = state.selectedFile !== null;
    const hasQuestion = elements.questionInput.value.trim().length >= 2;
    const canSubmit = hasImage && hasQuestion && !state.isLoading;

    elements.submitBtn.disabled = !canSubmit;
}

/**
 * Show loading overlay
 */
function showLoading() {
    state.isLoading = true;
    elements.loadingOverlay.style.display = 'flex';
    elements.submitBtn.querySelector('.btn-text').textContent = 'Processing...';
    elements.submitBtn.querySelector('.btn-loader').style.display = 'block';
    updateSubmitButton();
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    state.isLoading = false;
    elements.loadingOverlay.style.display = 'none';
    elements.submitBtn.querySelector('.btn-text').textContent = 'Get Answer';
    elements.submitBtn.querySelector('.btn-loader').style.display = 'none';
    updateSubmitButton();
}

/**
 * Format probability as percentage
 */
function formatProbability(prob) {
    return `${(prob * 100).toFixed(1)}%`;
}

// ============================================
// Image Handling
// ============================================

/**
 * Handle file selection
 */
function handleFileSelect(file) {
    const validation = validateFile(file);

    if (!validation.valid) {
        showStatus(validation.error, 'error');
        return;
    }

    state.selectedFile = file;
    hideStatus();

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        elements.imagePreview.src = e.target.result;
        elements.previewContainer.style.display = 'block';
        elements.dropZone.style.display = 'none';
    };
    reader.readAsDataURL(file);

    updateSubmitButton();
}

/**
 * Remove selected image
 */
function removeSelectedImage() {
    state.selectedFile = null;
    elements.imageInput.value = '';
    elements.imagePreview.src = '';
    elements.previewContainer.style.display = 'none';
    elements.dropZone.style.display = 'block';

    // Reset results
    elements.resultsPlaceholder.style.display = 'block';
    elements.resultsContent.style.display = 'none';

    updateSubmitButton();
}

// ============================================
// API Communication
// ============================================

/**
 * Send prediction request to API
 */
async function submitPrediction() {
    if (!state.selectedFile || elements.questionInput.value.trim().length < 2) {
        return;
    }

    showLoading();
    hideStatus();

    try {
        const formData = new FormData();
        formData.append('image', state.selectedFile);
        formData.append('question', elements.questionInput.value.trim());
        formData.append('top_k', CONFIG.topK);

        const response = await fetch(`${CONFIG.apiBaseUrl}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Prediction failed');
        }

        displayResults(result);

    } catch (error) {
        console.error('Prediction error:', error);

        // Check if it's a network error
        if (error.message.includes('Failed to fetch')) {
            showStatus('Cannot connect to server. Make sure the API is running at ' + CONFIG.apiBaseUrl, 'error');
        } else {
            showStatus(error.message, 'error');
        }
    } finally {
        hideLoading();
    }
}

/**
 * Display prediction results
 */
function displayResults(result) {
    // Hide placeholder, show results
    elements.resultsPlaceholder.style.display = 'none';
    elements.resultsContent.style.display = 'block';

    // Top answer
    elements.topAnswer.textContent = result.top_answer;

    // Confidence bar
    const confidencePercent = result.confidence * 100;
    elements.confidenceFill.style.width = `${confidencePercent}%`;
    elements.confidenceText.textContent = `Confidence: ${formatProbability(result.confidence)}`;

    // Alternative answers
    elements.altAnswersList.innerHTML = '';

    // Skip the first one (it's the top answer)
    const alternatives = result.answers.slice(1);

    alternatives.forEach(ans => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span class="alt-answer">${ans.answer}</span>
            <span class="alt-prob">${formatProbability(ans.probability)}</span>
        `;
        elements.altAnswersList.appendChild(li);
    });

    // Show success status
    showStatus('Prediction completed successfully!', 'success');
}

// ============================================
// Event Listeners
// ============================================

// Drop zone click
elements.dropZone.addEventListener('click', () => {
    elements.imageInput.click();
});

// File input change
elements.imageInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
    }
});

// Drag and drop events
elements.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.dropZone.classList.add('drag-over');
});

elements.dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    elements.dropZone.classList.remove('drag-over');
});

elements.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.dropZone.classList.remove('drag-over');

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

// Remove image button
elements.removeImage.addEventListener('click', removeSelectedImage);

// Question input
elements.questionInput.addEventListener('input', (e) => {
    const length = e.target.value.length;
    elements.charCount.textContent = `${length}/200`;
    updateSubmitButton();
});

// Enter key to submit
elements.questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !elements.submitBtn.disabled) {
        submitPrediction();
    }
});

// Example question buttons
elements.exampleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        elements.questionInput.value = btn.dataset.question;
        elements.charCount.textContent = `${btn.dataset.question.length}/200`;
        updateSubmitButton();
    });
});

// Submit button
elements.submitBtn.addEventListener('click', submitPrediction);

// ============================================
// Initialization
// ============================================

/**
 * Check API health on load
 */
async function checkApiHealth() {
    try {
        const response = await fetch(`${CONFIG.apiBaseUrl}/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            console.log('✓ API is healthy');
            if (data.model_loaded) {
                console.log('✓ Model is loaded');
            }
        }
    } catch (error) {
        console.warn('API health check failed:', error.message);
        showStatus('Warning: Cannot connect to API server. Please ensure the backend is running.', 'error');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    updateSubmitButton();
    checkApiHealth();

    console.log('VQA Frontend initialized');
    console.log('API URL:', CONFIG.apiBaseUrl);
});
