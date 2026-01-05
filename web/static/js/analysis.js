/**
 * Stock Pattern Predictor - Analysis Page JavaScript
 */

// Get symbol from URL path
const pathParts = window.location.pathname.split('/');
const symbol = pathParts[pathParts.length - 1] || pathParts[pathParts.length - 2];

// API Base URL
const API_BASE = '';

// Chart instance
let priceChart = null;

/**
 * Show loading state
 */
function showLoading() {
    document.getElementById('loadingState').style.display = 'block';
    document.getElementById('errorState').style.display = 'none';
    document.getElementById('contentState').style.display = 'none';
}

/**
 * Show error state
 */
function showError(message) {
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('errorState').style.display = 'block';
    document.getElementById('contentState').style.display = 'none';
    document.getElementById('errorMessage').textContent = message;
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
    // Default fallback
    return t('error_generic');
}

/**
 * Show content state
 */
function showContent() {
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('errorState').style.display = 'none';
    document.getElementById('contentState').style.display = 'block';
}

/**
 * Fetch analysis data from API
 */
async function fetchAnalysis() {
    console.log('[DEBUG] fetchAnalysis() called');
    console.log('[DEBUG] symbol:', symbol);

    showLoading();

    try {
        const url = `${API_BASE}/api/analyze?symbol=${symbol}&threshold=0.05&include_secondary=true`;
        console.log('[DEBUG] Fetching URL:', url);

        const response = await fetch(url);
        console.log('[DEBUG] Response status:', response.status);
        console.log('[DEBUG] Response ok:', response.ok);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            const errorMsg = errorData.detail || 'Failed to fetch analysis';
            console.log('[DEBUG] API error:', errorMsg);
            throw new Error(errorMsg);
        }

        const data = await response.json();
        console.log('[DEBUG] API response received');
        console.log('[DEBUG] data keys:', Object.keys(data));
        console.log('[DEBUG] data.symbol:', data.symbol);
        console.log('[DEBUG] data.current_price:', data.current_price);
        console.log('[DEBUG] data.landmarks count:', data.landmarks ? data.landmarks.length : 'undefined');
        console.log('[DEBUG] data.chart_data count:', data.chart_data ? data.chart_data.length : 'undefined');
        console.log('[DEBUG] data.multi_layer_stats:', data.multi_layer_stats);

        displayAnalysis(data);

    } catch (error) {
        console.log('[DEBUG] fetchAnalysis error:', error);
        console.log('[DEBUG] error.message:', error.message);
        console.log('[DEBUG] error.stack:', error.stack);
        showError(translateError(error.message));
    }
}

/**
 * Display analysis data
 */
function displayAnalysis(data) {
    console.log('[DEBUG] displayAnalysis() called');
    console.log('[DEBUG] data:', data);

    try {
        console.log('[DEBUG] Step 1: Update stock info');

        // Update stock info
        const stockSymbolEl = document.getElementById('stockSymbol');
        console.log('[DEBUG] stockSymbol element:', stockSymbolEl);
        stockSymbolEl.textContent = data.symbol + (data.name && data.name !== data.symbol ? ' - ' + data.name : '');

        const currentPriceEl = document.getElementById('currentPrice');
        console.log('[DEBUG] currentPrice:', data.current_price);
        currentPriceEl.textContent = data.current_price.toFixed(2);

        document.getElementById('lastWeek').textContent = data.last_week;
        document.getElementById('lastDate').textContent = data.last_date;
        document.getElementById('landmarksCount').textContent = data.landmarks.length;
        // threshold element doesn't exist in the template, skipping
        // document.getElementById('threshold').textContent = (data.threshold * 100).toFixed(1) + '%';

        console.log('[DEBUG] Step 2: Update multi-layer stats');

        // Update multi-layer stats if available
        if (data.multi_layer_stats) {
            console.log('[DEBUG] multi_layer_stats:', data.multi_layer_stats);
            document.getElementById('filterRate').textContent = data.multi_layer_stats.filter_rate;
            document.getElementById('mlLayer1').textContent = data.multi_layer_stats.layer1_count;
            document.getElementById('mlLayer2').textContent = data.multi_layer_stats.layer2_count;
            document.getElementById('mlLayer3').textContent = data.multi_layer_stats.layer3_count;
            document.getElementById('mlFinal').textContent = data.multi_layer_stats.final_count;
        } else {
            console.log('[DEBUG] No multi_layer_stats in response');
        }

        console.log('[DEBUG] Step 3: Create chart');
        console.log('[DEBUG] chart_data length:', data.chart_data.length);
        console.log('[DEBUG] landmarks length:', data.landmarks.length);

        // Create chart
        createChart(data.chart_data, data.landmarks);

        console.log('[DEBUG] Step 4: Populate landmarks table');

        // Populate landmarks table
        populateLandmarksTable(data.landmarks);

        console.log('[DEBUG] Step 5: Show content');

        showContent();

        console.log('[DEBUG] displayAnalysis() completed successfully');

    } catch (error) {
        console.error('[ERROR] displayAnalysis error:', error);
        console.error('[ERROR] error.message:', error.message);
        console.error('[ERROR] error.stack:', error.stack);
        showError('显示数据时出错: ' + error.message);
    }
}

/**
 * Create price chart with landmarks
 */
function createChart(chartData, landmarks) {
    console.log('[DEBUG] createChart() called');
    console.log('[DEBUG] chartData length:', chartData.length);
    console.log('[DEBUG] landmarks length:', landmarks.length);

    try {
        const ctx = document.getElementById('priceChart');
        console.log('[DEBUG] Canvas element:', ctx);

        if (!ctx) {
            throw new Error('Canvas element not found');
        }

        const ctx2d = ctx.getContext('2d');
        console.log('[DEBUG] Canvas 2d context:', ctx2d);

        // Prepare data - use chart data directly (should be weekly data)
        const labels = chartData.map(d => d.date);
        const prices = chartData.map(d => d.close);
        console.log('[DEBUG] labels length:', labels.length);
        console.log('[DEBUG] prices length:', prices.length);
        console.log('[DEBUG] Sample label:', labels[0]);
        console.log('[DEBUG] Sample price:', prices[0]);

        // Prepare landmark annotations - separate primary and secondary
        const primaryLowPoints = [];
        const primaryHighPoints = [];
        const secondaryLowPoints = [];
        const secondaryHighPoints = [];

        // Create a date lookup for O(1) matching
        const dateToIndex = {};
        labels.forEach((date, idx) => {
            dateToIndex[date] = idx;
        });

        console.log('[DEBUG] Matching landmarks to chart dates...');
        landmarks.forEach(lm => {
            const idx = dateToIndex[lm.date];
            if (idx !== undefined) {
                const isPrimary = lm.level === 'primary' || !lm.level;
                const isLow = lm.type === 'low';

                if (isPrimary && isLow) {
                    primaryLowPoints[idx] = lm.price;
                } else if (isPrimary && !isLow) {
                    primaryHighPoints[idx] = lm.price;
                } else if (!isPrimary && isLow) {
                    secondaryLowPoints[idx] = lm.price;
                } else if (!isPrimary && !isLow) {
                    secondaryHighPoints[idx] = lm.price;
                }
            }
        });

        console.log('[DEBUG] primaryLowPoints:', primaryLowPoints.filter(p => p !== undefined).length);
        console.log('[DEBUG] primaryHighPoints:', primaryHighPoints.filter(p => p !== undefined).length);
        console.log('[DEBUG] secondaryLowPoints:', secondaryLowPoints.filter(p => p !== undefined).length);
        console.log('[DEBUG] secondaryHighPoints:', secondaryHighPoints.filter(p => p !== undefined).length);

        // Destroy existing chart
        if (priceChart) {
            console.log('[DEBUG] Destroying existing chart');
            priceChart.destroy();
        }

        console.log('[DEBUG] Creating new Chart instance...');

        // Create new chart
        priceChart = new Chart(ctx2d, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: t('chart_legend_price'),
                        data: prices,
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        fill: true,
                        tension: 0.1,
                        order: 3
                    },
                    {
                        label: t('chart_legend_low'),
                        data: primaryLowPoints.map((p, i) => p !== undefined ? p : NaN),
                        borderColor: 'rgba(40, 167, 69, 1)',
                        backgroundColor: 'rgba(40, 167, 69, 1)',
                        pointStyle: 'triangle',
                        pointRadius: 10,
                        showLine: false,
                        order: 1
                    },
                    {
                        label: t('chart_legend_high'),
                        data: primaryHighPoints.map((p, i) => p !== undefined ? p : NaN),
                        borderColor: 'rgba(220, 53, 69, 1)',
                        backgroundColor: 'rgba(220, 53, 69, 1)',
                        pointStyle: 'triangle',
                        pointRadius: 10,
                        rotation: 180,
                        showLine: false,
                        order: 1
                    },
                    {
                        label: t('chart_legend_low') + ' (Minor)',
                        data: secondaryLowPoints.map((p, i) => p !== undefined ? p : NaN),
                        borderColor: 'rgba(40, 167, 69, 0.6)',
                        backgroundColor: 'rgba(40, 167, 69, 0.6)',
                        pointStyle: 'triangle',
                        pointRadius: 5,
                        showLine: false,
                        order: 2
                    },
                    {
                        label: t('chart_legend_high') + ' (Minor)',
                        data: secondaryHighPoints.map((p, i) => p !== undefined ? p : NaN),
                        borderColor: 'rgba(220, 53, 69, 0.6)',
                        backgroundColor: 'rgba(220, 53, 69, 0.6)',
                        pointStyle: 'triangle',
                        pointRadius: 5,
                        rotation: 180,
                        showLine: false,
                        order: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        enabled: true,
                        callbacks: {
                            title: function(context) {
                                const idx = context[0].dataIndex;
                                return labels[idx];
                            },
                            label: function(context) {
                                if (context.datasetIndex === 0) {
                                    return `Close: ¥${context.parsed.y.toFixed(2)}`;
                                } else {
                                    const price = context.raw;
                                    if (price && !isNaN(price)) {
                                        return `${context.dataset.label}: ¥${price.toFixed(2)}`;
                                    }
                                    return null;
                                }
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Week'
                        },
                        ticks: {
                            maxTicksLimit: 15,
                            maxRotation: 45,
                            minRotation: 45
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Price (¥)'
                        }
                    }
                }
            }
        });

        console.log('[DEBUG] Chart created successfully');

    } catch (error) {
        console.error('[ERROR] createChart error:', error);
        console.error('[ERROR] error.message:', error.message);
        console.error('[ERROR] error.stack:', error.stack);
        throw error; // Re-throw to let displayAnalysis handle it
    }
}

/**
 * Populate landmarks table
 */
function populateLandmarksTable(landmarks) {
    const tbody = document.getElementById('landmarksTable');

    if (landmarks.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" class="text-center">${t('table_no_data')}</td></tr>`;
        return;
    }

    // Sort by date descending
    const sortedLandmarks = [...landmarks].sort((a, b) => new Date(b.date) - new Date(a.date));

    tbody.innerHTML = sortedLandmarks.map(lm => {
        const level = lm.level || 'primary';
        const levelClass = level === 'primary' ? 'bg-dark' : 'bg-secondary';
        const levelLabel = level === 'primary' ? 'Major' : 'Minor';

        return `
        <tr class="${level === 'secondary' ? 'table-secondary' : ''}">
            <td>
                <span class="badge ${lm.type === 'low' ? 'bg-success' : 'bg-danger'}">
                    ${lm.type.toUpperCase()}
                </span>
            </td>
            <td>${lm.date}</td>
            <td>¥${lm.price.toFixed(2)}</td>
            <td>${lm.weekly_index}</td>
            <td>
                <span class="badge ${levelClass}">
                    ${levelLabel}
                </span>
            </td>
            <td><small class="text-muted">${lm.reasons ? lm.reasons.join(', ') : '-'}</small></td>
        </tr>
        `;
    }).join('');
}

/**
 * Refresh analysis
 */
function refreshAnalysis() {
    const btn = document.querySelector('button[onclick="refreshAnalysis()"]');
    const spinner = document.getElementById('refreshSpinner');
    const text = document.getElementById('refreshText');

    btn.disabled = true;
    spinner.classList.remove('d-none');
    text.textContent = 'Loading...';

    fetchAnalysis().finally(() => {
        btn.disabled = false;
        spinner.classList.add('d-none');
        text.textContent = t('btn_refresh');
        text.setAttribute('data-i18n', 'btn_refresh');
    });
}

/**
 * Initialize on page load
 */
document.addEventListener('DOMContentLoaded', () => {
    fetchAnalysis();
    setupMultiLayerSliders();
    setupQuickSearch();
});

/**
 * Setup quick search functionality
 */
function setupQuickSearch() {
    const searchInput = document.getElementById('quickSearchInput');
    const autocompleteDiv = document.getElementById('quickSearchAutocomplete');
    let searchTimeout = null;

    // Handle input changes
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();

        // Clear previous timeout
        if (searchTimeout) {
            clearTimeout(searchTimeout);
        }

        // Hide autocomplete if query is too short
        if (query.length < 1) {
            hideAutocomplete();
            return;
        }

        // Debounce search
        searchTimeout = setTimeout(async () => {
            const results = await searchStocks(query);
            showAutocomplete(results);
        }, 300);
    });

    // Handle keyboard navigation
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const query = searchInput.value.trim();
            if (query) {
                // Try to extract 6-digit stock code
                const codeMatch = query.match(/\d{6}/);
                if (codeMatch) {
                    navigateToAnalysis(codeMatch[0]);
                } else {
                    // Search and use first result
                    searchStocks(query).then(results => {
                        if (results.length > 0) {
                            navigateToAnalysis(results[0].code);
                        }
                    });
                }
                hideAutocomplete();
            }
        } else if (e.key === 'Escape') {
            hideAutocomplete();
        }
    });

    // Hide autocomplete when clicking outside
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !autocompleteDiv.contains(e.target)) {
            hideAutocomplete();
        }
    });
}

/**
 * Search stocks by code/pinyin/name
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
    const autocompleteDiv = document.getElementById('quickSearchAutocomplete');

    // Remove existing dropdown
    hideAutocomplete();

    if (!results || results.length === 0) {
        return;
    }

    // Create dropdown
    const dropdown = document.createElement('div');
    dropdown.className = 'list-group';

    results.forEach(result => {
        const item = document.createElement('button');
        item.className = 'list-group-item list-group-item-action';
        item.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <strong>${result.code}</strong>
                    <span class="ms-2">${result.name}</span>
                    <span class="badge bg-secondary ms-2">${result.exchange}</span>
                </div>
                ${result.match_type ? `<small class="text-muted">${result.match_type}</small>` : ''}
            </div>
        `;

        item.addEventListener('click', () => {
            navigateToAnalysis(result.code);
            hideAutocomplete();
        });

        dropdown.appendChild(item);
    });

    autocompleteDiv.appendChild(dropdown);
}

/**
 * Hide autocomplete dropdown
 */
function hideAutocomplete() {
    const autocompleteDiv = document.getElementById('quickSearchAutocomplete');
    autocompleteDiv.innerHTML = '';
}

/**
 * Navigate to analysis page for a different stock
 */
function navigateToAnalysis(code) {
    // Ensure code is 6 digits
    const paddedCode = code.toString().padStart(6, '0');
    window.location.href = `/analysis/${paddedCode}`;
}

/**
 * Setup multi-layer filter sliders
 */
function setupMultiLayerSliders() {
    const sliders = [
        { id: 'layer1_threshold', display: 'l1_val', suffix: '%' },
        { id: 'layer2_min_freq', display: 'l2_freq_val', suffix: '' },
        { id: 'layer2_min_dev', display: 'l2_dev_val', suffix: '%' },
        { id: 'layer3_trend_str', display: 'l3_str_val', suffix: '', format: v => (v/100).toFixed(2) },
        { id: 'layer4_same_type', display: 'l4_same_val', suffix: '' },
        { id: 'layer4_alt', display: 'l4_alt_val', suffix: '' }
    ];

    sliders.forEach(slider => {
        const input = document.getElementById(slider.id);
        const display = document.getElementById(slider.display);

        if (input && display) {
            input.addEventListener('input', function() {
                const val = this.value;
                if (slider.format) {
                    display.textContent = slider.format(val);
                } else {
                    display.textContent = val + slider.suffix;
                }
            });
        }
    });
}

/**
 * Run multi-layer detection
 */
async function runMultiLayerDetection() {
    const btn = document.querySelector('button[onclick="runMultiLayerDetection()"]');
    const spinner = document.getElementById('mlUpdateSpinner');
    const text = document.getElementById('mlUpdateText');
    const resultsDiv = document.getElementById('multiLayerResults');

    // Show loading
    btn.disabled = true;
    spinner.classList.remove('d-none');
    text.textContent = '检测中...';

    try {
        const params = {
            symbol: symbol,
            layer1_threshold: parseInt(document.getElementById('layer1_threshold').value),
            layer2_min_freq: parseInt(document.getElementById('layer2_min_freq').value),
            layer2_min_dev: parseInt(document.getElementById('layer2_min_dev').value),
            layer3_trend_str: parseInt(document.getElementById('layer3_trend_str').value),
            layer4_same_type: parseInt(document.getElementById('layer4_same_type').value),
            layer4_alt: parseInt(document.getElementById('layer4_alt').value)
        };

        const response = await fetch('/api/multi-layer-detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || 'Multi-layer detection failed');
        }

        const data = await response.json();

        // Update statistics
        document.getElementById('ml_layer1').textContent = data.stats.layer1_count;
        document.getElementById('ml_layer2').textContent = data.stats.layer2_count;
        document.getElementById('ml_layer3').textContent = data.stats.layer3_count;
        document.getElementById('ml_final').textContent = data.stats.final_count;

        // Update plot
        document.getElementById('mlPlot').src = 'data:image/png;base64,' + data.plot;

        // Show results
        resultsDiv.style.display = 'block';

        // Scroll to results
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (error) {
        console.error('Multi-layer detection error:', error);
        alert('检测失败: ' + error.message);
    } finally {
        btn.disabled = false;
        spinner.classList.add('d-none');
        text.textContent = '运行检测';
    }
}
