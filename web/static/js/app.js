/**
 * Stock Pattern Predictor - Frontend JavaScript
 */

// API Base URL
const API_BASE = '';

// DOM Elements
const searchForm = document.getElementById('searchForm');
const symbolInput = document.getElementById('symbolInput');
const resultsSection = document.getElementById('resultsSection');
const loadingState = document.getElementById('loadingState');
const errorState = document.getElementById('errorState');
const successState = document.getElementById('successState');
const errorMessage = document.getElementById('errorMessage');

// Autocomplete
let searchTimeout = null;
let autocompleteResults = [];

/**
 * Quick search from example buttons
 */
function quickSearch(symbol) {
    symbolInput.value = symbol;
    searchForm.dispatchEvent(new Event('submit'));
}

/**
 * Show loading state
 */
function showLoading() {
    resultsSection.style.display = 'block';
    loadingState.style.display = 'block';
    errorState.style.display = 'none';
    successState.style.display = 'none';

    // Add loading class to button
    const submitBtn = searchForm.querySelector('button[type="submit"]');
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;
}

/**
 * Show error state
 */
function showError(message) {
    resultsSection.style.display = 'block';
    loadingState.style.display = 'none';
    errorState.style.display = 'block';
    successState.style.display = 'none';
    errorMessage.textContent = message;

    // Remove loading class from button
    const submitBtn = searchForm.querySelector('button[type="submit"]');
    submitBtn.classList.remove('loading');
    submitBtn.disabled = false;
}

/**
 * Show success state with results
 */
function showSuccess(data) {
    resultsSection.style.display = 'block';
    loadingState.style.display = 'none';
    errorState.style.display = 'none';
    successState.style.display = 'block';

    // Populate results
    document.getElementById('resultSymbol').textContent =
        data.symbol + (data.name && data.name !== data.symbol ? ' - ' + data.name : '');
    document.getElementById('resultCurrentPrice').textContent = '¥' + data.current_price.toFixed(2);
    document.getElementById('resultType').textContent = data.landmark_type.toUpperCase();
    document.getElementById('resultWeek').textContent = data.predicted_week;
    document.getElementById('resultDateRange').textContent = data.predicted_week_start + ' to ' + data.predicted_week_end;

    // Format weeks away
    const weeksAway = data.weeks_to_prediction;
    const monthsAway = (weeksAway / 4).toFixed(1);
    let weeksAwayText = weeksAway > 0
        ? `+${weeksAway} weeks (~${monthsAway} months)`
        : `${weeksAway} weeks (${Math.abs(monthsAway)} months ago)`;
    document.getElementById('resultWeeksAway').textContent = weeksAwayText;

    // Confidence with color and translation
    const confidenceEl = document.getElementById('resultConfidence');
    const confidenceKey = 'confidence_' + data.confidence.toLowerCase();
    confidenceEl.textContent = t(confidenceKey);
    confidenceEl.className = ''; // Reset classes
    if (data.confidence === 'HIGH') {
        confidenceEl.classList.add('confidence-high');
    } else if (data.confidence === 'MEDIUM') {
        confidenceEl.classList.add('confidence-medium');
    } else {
        confidenceEl.classList.add('confidence-low');
    }

    // Pattern details
    document.getElementById('resultPattern').textContent = data.pattern_expression;
    document.getElementById('resultFrequency').textContent = data.pattern_frequency;
    document.getElementById('resultSequence').textContent = '[' + data.sequence.join(', ') + ']';

    // Display all pattern expressions if available
    if (data.pattern_expressions && Object.keys(data.pattern_expressions).length > 0) {
        const patternDetailsEl = document.getElementById('resultPatternDetails');
        if (patternDetailsEl) {
            let html = '<div class="mt-3"><strong>All Pattern Expressions (→ Week ' + data.predicted_week + '):</strong><div class="row mt-2">';

            for (const [ptype, info] of Object.entries(data.pattern_expressions)) {
                html += `
                    <div class="col-md-6 mb-3">
                        <div class="card">
                            <div class="card-body p-2">
                                <h6 class="card-title mb-1">${ptype}</h6>
                                <small class="text-muted">${info.count} expressions → Week ${data.predicted_week}</small>
                                <div class="mt-1" style="max-height: 120px; overflow-y: auto; font-size: 0.85rem;">
                                    ${info.examples.map(e => `<div><code>${e}</code> = <strong>${data.predicted_week}</strong></div>`).join('')}
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }

            html += '</div></div>';
            patternDetailsEl.innerHTML = html;
        }
    }

    // Update view details button
    const viewDetailsBtn = document.getElementById('viewDetailsBtn');
    viewDetailsBtn.href = `/analysis/${data.symbol}`;

    // Remove loading class from button
    const submitBtn = searchForm.querySelector('button[type="submit"]');
    submitBtn.classList.remove('loading');
    submitBtn.disabled = false;
}

/**
 * Translate error message from backend
 */
function translateError(message) {
    // Check for common error patterns and translate them
    if (message.includes('not found')) {
        const match = message.match(/Stock (\w+) not found/);
        if (match) {
            return t('error_stock_not_found').replace('{symbol}', match[1]);
        }
    }
    if (message.includes('No landmarks found')) {
        const match = message.match(/No landmarks found for (\w+) with threshold ([\d.]+)/);
        if (match) {
            return t('error_no_landmarks')
                .replace('{symbol}', match[1])
                .replace('{threshold}', (parseFloat(match[2]) * 100).toFixed(0) + '%');
        }
    }
    if (message.includes('Insufficient')) {
        const match = message.match(/Insufficient.*landmarks.*\((\d+), got (\d+)\)/);
        if (match) {
            return t('error_insufficient_landmarks')
                .replace('{symbol}', match[1])
                .replace('{count}', match[2]);
        }
    }
    if (message.includes('No future predictions')) {
        const match = message.match(/No future predictions found for (\w+)/);
        if (match) {
            return t('error_no_predictions').replace('{symbol}', match[1]);
        }
    }
    // Default fallback
    return t('error_generic');
}

/**
 * Fetch prediction from API
 */
async function fetchPrediction(symbol) {
    const url = `${API_BASE}/api/predict?symbol=${symbol}&landmark_type=low&threshold=0.05`;

    try {
        const response = await fetch(url);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            const errorMsg = errorData.detail || 'Failed to fetch prediction';
            throw new Error(errorMsg);
        }

        const data = await response.json();
        showSuccess(data);

    } catch (error) {
        showError(translateError(error.message));
    }
}

/**
 * Search stocks by pinyin, code, or name
 */
async function searchStocks(query) {
    if (!query || query.length < 1) {
        return [];
    }

    try {
        const url = `${API_BASE}/api/stocks/search?q=${encodeURIComponent(query)}&limit=10`;
        const response = await fetch(url);

        if (!response.ok) {
            return [];
        }

        const data = await response.json();
        return data.results || [];
    } catch (error) {
        console.error('Search error:', error);
        return [];
    }
}

/**
 * Show autocomplete dropdown
 */
function showAutocomplete(results) {
    // Remove existing dropdown
    hideAutocomplete();

    if (!results || results.length === 0) {
        return;
    }

    // Create dropdown
    const dropdown = document.createElement('div');
    dropdown.className = 'autocomplete-dropdown';
    dropdown.id = 'autocompleteDropdown';

    // Add results
    results.forEach(result => {
        const item = document.createElement('div');
        item.className = 'autocomplete-item';
        item.innerHTML = `
            <span class="stock-code">${result.code}</span>
            <span class="stock-name">${result.name}</span>
            <span class="stock-exchange">${result.exchange}</span>
        `;
        item.addEventListener('click', () => {
            symbolInput.value = result.code;
            hideAutocomplete();
            searchForm.dispatchEvent(new Event('submit'));
        });
        dropdown.appendChild(item);
    });

    // Position dropdown
    const inputRect = symbolInput.getBoundingClientRect();
    dropdown.style.top = (inputRect.bottom + window.scrollY) + 'px';
    dropdown.style.left = inputRect.left + 'px';
    dropdown.style.width = inputRect.width + 'px';

    document.body.appendChild(dropdown);
}

/**
 * Hide autocomplete dropdown
 */
function hideAutocomplete() {
    const dropdown = document.getElementById('autocompleteDropdown');
    if (dropdown) {
        dropdown.remove();
    }
}

/**
 * Handle form submission
 */
searchForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    let symbol = symbolInput.value.trim();

    // Validate input
    if (!symbol) {
        showError(t('search_required'));
        return;
    }

    // If input is not 6 digits, try to search for stock
    if (!/^\d{6}$/.test(symbol)) {
        const results = await searchStocks(symbol);
        if (results.length === 1) {
            // Single match, use it
            symbol = results[0].code;
        } else if (results.length > 1) {
            // Multiple matches, show dropdown and wait for selection
            showError(t('search_select_from_matches').replace('{count}', results.length));
            showAutocomplete(results);
            return;
        } else {
            // No matches
            showError(t('search_not_found'));
            return;
        }
    }

    // Pad with leading zeros
    symbol = symbol.padStart(6, '0');

    hideAutocomplete();
    showLoading();
    await fetchPrediction(symbol);
});

/**
 * Handle input changes with autocomplete
 */
symbolInput.addEventListener('input', async (e) => {
    let value = e.target.value.trim();

    // Hide existing dropdown
    hideAutocomplete();

    // Remove validation classes
    symbolInput.classList.remove('is-valid', 'is-invalid');

    // If empty, nothing to do
    if (!value) {
        return;
    }

    // If already 6 digits, validate
    if (/^\d{6}$/.test(value)) {
        symbolInput.classList.add('is-valid');
        return;
    }

    // Don't show autocomplete for pure digits (user is typing code)
    if (/^\d+$/.test(value)) {
        return;
    }

    // For non-digit input (pinyin), show autocomplete
    // Debounce the search
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(async () => {
        const results = await searchStocks(value);
        if (results.length > 0) {
            showAutocomplete(results);
        }
    }, 300);
});

/**
 * Handle Enter key in input
 */
symbolInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        hideAutocomplete();
        searchForm.dispatchEvent(new Event('submit'));
    }
});

/**
 * Hide autocomplete when clicking outside
 */
document.addEventListener('click', (e) => {
    if (!symbolInput.contains(e.target)) {
        hideAutocomplete();
    }
});

/**
 * Initialize tooltips and other UI elements
 */
document.addEventListener('DOMContentLoaded', () => {
    // Focus on input on page load
    symbolInput.focus();
});
