/**
 * Translations for Stock Pattern Predictor
 * Supports English (en) and Chinese (zh)
 */

const translations = {
    en: {
        // Navigation
        nav_home: "Home",
        nav_analysis: "Analysis",
        nav_about: "About",

        // Hero Section
        hero_title: "A-Share Stock Pattern Prediction",
        hero_subtitle: "Predict WHEN landmark low/high points will occur using time-based sequence analysis",

        // Search Form
        search_label: "Stock Code (6 digits)",
        search_placeholder: "Enter stock code, e.g., 601933",
        search_button: "Analyze",
        search_help: "Example: 601933, 000001, 600036, 000858<br>Or pinyin: jd, ynsh, wly",
        search_required: "Please enter a stock code or pinyin",
        search_invalid: "Stock code must be 6 digits (e.g., 601933)",
        search_select_from_matches: "Found {count} matches. Please select from the list.",
        search_not_found: "No matching stock found. Try a different search term.",

        // Error Messages
        error_stock_not_found: "Stock {symbol} not found",
        error_no_landmarks: "No landmarks found for {symbol} with threshold {threshold}",
        error_insufficient_landmarks: "Insufficient landmarks for {symbol} (need 3+, got {count})",
        error_no_predictions: "No future predictions found for {symbol}",
        error_generic: "An error occurred. Please try again.",

        // Results Section
        results_title: "Prediction Results",
        results_loading: "Analyzing stock pattern...",
        error_title: "Error",
        btn_view_details: "View Full Analysis →",
        btn_back: "← Back to Search",
        btn_refresh: "Refresh",

        // Stock Info
        info_symbol: "Symbol",
        info_current_price: "Current Price",
        info_prediction_type: "Prediction Type",
        info_predicted_week: "Predicted Week",
        info_date_range: "Date Range",
        info_weeks_away: "Weeks Away",
        info_confidence: "Confidence",
        info_last_week: "Last Data Week",
        info_last_date: "Last Date",
        info_landmarks_count: "Landmarks Found",
        info_threshold: "ZigZag Threshold",

        // Pattern Details
        pattern_expression: "Pattern Expression",
        pattern_frequency: "Pattern Frequency",
        sequence_label: "Sequence (Week Indices)",

        // Confidence Levels
        confidence_high: "HIGH",
        confidence_medium: "MEDIUM",
        confidence_low: "LOW",

        // Analysis Page
        analysis_title: "Stock Analysis",
        chart_title: "Price Chart with Landmarks",
        chart_legend_price: "Close Price",
        chart_legend_low: "Low Landmarks",
        chart_legend_high: "High Landmarks",
        chart_hint: "Green triangles = Low landmarks | Red triangles = High landmarks",
        table_title: "Detected Landmarks",
        table_type: "Type",
        table_date: "Date",
        table_price: "Price",
        table_week_index: "Week Index",
        table_confidence: "Confidence",
        table_signals: "Signals",
        table_no_data: "No landmarks found",

        // Footer
        footer_text: "Stock Pattern Predictor | Time-Based Sequence Analysis",
        footer_link: "How it works",

        // Examples
        examples_title: "Quick Examples:",

        // Loading
        loading_analysis: "Loading analysis data...",

        // About Page
        page_title_about: "About - Stock Pattern Predictor",
        about_title: "How Stock Pattern Prediction Works",
        about_subtitle: "Understanding time-based sequence analysis for stock market landmarks",
        section_what_is_title: "What is Time-Based Sequence Analysis?",
        section_what_is_p1: "Traditional technical analysis focuses on PRICE patterns (head & shoulders, double bottom, etc.). Our approach is different: we analyze TIME patterns.",
        section_what_is_p2: "We ask: WHEN do landmark points (significant lows or highs) tend to occur, measured in week indices from the stock's listing date?",
        example_title: "Example:",
        section_what_is_example: "If a stock has significant low points at weeks 52, 366, 680, 994... we look for mathematical patterns in these numbers to predict the NEXT low point.",
        section_how_title: "How It Works: Step by Step",
        step1_title: "Collect Historical Data",
        step1_desc: "Download weekly OHLCV data from stock's IPO to present.",
        step2_title: "Detect Landmarks",
        step2_desc: "Use ZigZag algorithm (5% threshold) to identify significant price reversals.",
        step3_title: "Extract Week Indices",
        step3_desc: "Convert landmark dates to week numbers from IPO (e.g., week 366).",
        step4_title: "Find Mathematical Patterns",
        step4_desc: "Analyze sequences for arithmetic/geometric patterns (e.g., 52, 366, 680...).",
        step5_title: "Calculate Next Landmark",
        step5_desc: "Apply pattern to predict next occurrence week number.",
        step6_title: "Convert to Date Range",
        step6_desc: "Translate week number into calendar date range.",
        section_pattern_title: "Pattern Example: 601933",
        table_col_landmark: "Landmark",
        table_col_week: "Week Index",
        table_col_gap: "Gap",
        table_col_pattern: "Pattern",
        pattern_base: "Base",
        pattern_2nd: "2nd point",
        pattern_314: "314-week interval confirmed!",
        pattern_next: "Next: 680 + 314 = 994",
        pattern_note: "Note: Pattern frequency indicates how many times this interval has repeated historically.",
        section_week_title: "Understanding Week Indices",
        section_week_p1: "Week indices count the number of weeks from the stock's IPO (initial public offering):",
        section_week_li1: "Week 0 = First week of trading (IPO week)",
        section_week_li2: "Week 52 = Approximately 1 year after IPO",
        section_week_li3: "Week 640 = Approximately 12.3 years after IPO",
        section_week_p2: "By using week indices instead of calendar dates, we can discover mathematical patterns that repeat over time, regardless of market conditions, news events, or economic cycles.",
        section_confidence_title: "Confidence Levels Explained",
        confidence_high: "HIGH",
        confidence_high_desc: "Pattern frequency ≥ 3. Strong repeating pattern detected.",
        section_disclaimer_title: "Important Disclaimers",
        disclaimer_not_financial_advice: "⚠️ This is NOT Financial Advice",
        disclaimer_for_educational: "This tool is for educational and research purposes only",
        disclaimer_past_performance: "Past patterns do not guarantee future results",
        disclaimer_do_research: "Always do your own research before making investment decisions",
        disclaimer_consult_professional: "Consult a qualified financial advisor for investment guidance",
        section_faq_title: "Frequently Asked Questions",
        faq1_q: "What is the ZigZag algorithm?",
        faq1_a: "The ZigZag algorithm filters out minor price movements below a threshold (default 5%). It only identifies significant trend reversals, helping us focus on meaningful landmark points rather than noise.",
        faq2_q: "Why use weekly data instead of daily?",
        faq2_a: "Weekly data smooths out daily noise and reduces the number of data points, making patterns more visible. Landmark points typically persist over multiple days, so weekly aggregation preserves the signal while reducing noise.",
        faq3_q: "How accurate are these predictions?",
        faq3_a: "Accuracy varies significantly between stocks and depends on pattern strength. Stocks with longer histories and more consistent patterns tend to have higher confidence predictions. We are working on tracking historical accuracy to provide statistics.",
        faq4_q: "Can I use this for day trading?",
        faq4_a: "No. This tool predicts landmark points on a weekly basis, not daily price movements. It's designed for longer-term analysis, not short-term trading. The predictions indicate a week range when a landmark might occur, not a specific day or price.",
        cta_title: "Ready to Try?",
        cta_subtitle: "Search for any A-share stock to see its pattern predictions",
        cta_button: "Start Analyzing Stocks",

        // History Page
        page_title_history: "History - Stock Pattern Predictor",
        history_title: "Prediction History & Validation",
        history_coming_soon_title: "Feature Coming Soon",
        history_coming_soon_text: "Historical prediction tracking and validation is under development. This page will show past predictions, their accuracy, and statistical analysis of prediction performance over time.",
        history_coming_soon_detail: "<strong>Planned Features:</strong><br>• Database of all historical predictions made<br>• Actual outcome tracking (did the landmark occur?)<br>• Accuracy statistics by confidence level<br>• Pattern performance analysis<br>• Export prediction history to CSV",
        history_table_title: "Historical Predictions",
        history_no_data_title: "No Historical Data Yet",
        history_no_data_text: "Prediction tracking will begin once the feature is implemented. Check back later for updates!",
        hist_table_date_made: "Date Made",
        hist_table_symbol: "Symbol",
        hist_table_type: "Type",
        hist_table_predicted_week: "Predicted Week",
        hist_table_predicted_range: "Predicted Range",
        hist_table_actual_week: "Actual Week",
        hist_table_result: "Result",
        hist_table_confidence: "Confidence",
        stat_total_predictions: "Total Predictions",
        stat_accuracy: "Overall Accuracy",
        stat_high_confidence_accuracy: "High Confidence Accuracy"
    },
    zh: {
        // Navigation
        nav_home: "首页",
        nav_analysis: "分析",
        nav_about: "关于",

        // Hero Section
        hero_title: "A股形态预测",
        hero_subtitle: "基于时间序列分析，预测关键低点/高点出现时间",

        // Search Form
        search_label: "股票代码（6位数字）",
        search_placeholder: "请输入股票代码，如：601933",
        search_button: "分析",
        search_help: "示例：601933、000001、600036、000858<br>或拼音：jd、ynsh、wly",
        search_required: "请输入股票代码或拼音",
        search_invalid: "股票代码必须是6位数字（如：601933）",
        search_select_from_matches: "找到 {count} 个匹配结果，请从列表中选择。",
        search_not_found: "未找到匹配的股票，请尝试其他搜索词。",

        // Error Messages
        error_stock_not_found: "未找到股票 {symbol}",
        error_no_landmarks: "在阈值 {threshold} 下未找到 {symbol} 的形态点",
        error_insufficient_landmarks: "{symbol} 的形态点不足（需要3个以上，当前{count}个）",
        error_no_predictions: "未找到 {symbol} 的未来预测",
        error_generic: "发生错误，请重试。",

        // Results Section
        results_title: "预测结果",
        results_loading: "正在分析股票形态...",
        error_title: "错误",
        btn_view_details: "查看完整分析 →",
        btn_back: "← 返回搜索",
        btn_refresh: "刷新",

        // Stock Info
        info_symbol: "股票代码",
        info_current_price: "当前价格",
        info_prediction_type: "预测类型",
        info_predicted_week: "预测周",
        info_date_range: "日期范围",
        info_weeks_away: "距离周数",
        info_confidence: "置信度",
        info_last_week: "最新数据周",
        info_last_date: "最新日期",
        info_landmarks_count: "发现形态点",
        info_threshold: "ZigZag阈值",

        // Pattern Details
        pattern_expression: "形态表达式",
        pattern_frequency: "形态频率",
        sequence_label: "序列（周索引）",

        // Confidence Levels
        confidence_high: "高",
        confidence_medium: "中",
        confidence_low: "低",

        // Analysis Page
        analysis_title: "股票分析",
        chart_title: "价格走势与形态点",
        chart_legend_price: "收盘价",
        chart_legend_low: "低点标记",
        chart_legend_high: "高点标记",
        chart_hint: "绿色三角 = 低点 | 红色三角 = 高点",
        table_title: "检测到的形态点",
        table_type: "类型",
        table_date: "日期",
        table_price: "价格",
        table_week_index: "周索引",
        table_confidence: "置信度",
        table_signals: "信号",
        table_no_data: "未发现形态点",

        // Footer
        footer_text: "股票形态预测 | 基于时间序列分析",
        footer_link: "使用说明",

        // Examples
        examples_title: "快速示例：",

        // Loading
        loading_analysis: "正在加载分析数据...",

        // About Page
        page_title_about: "关于 - 股票形态预测",
        about_title: "股票形态预测原理",
        about_subtitle: "理解基于时间序列分析的股市形态点预测",
        section_what_is_title: "什么是基于时间的序列分析？",
        section_what_is_p1: "传统技术分析关注价格形态（头肩顶、双底等）。我们的方法不同：我们分析时间形态。",
        section_what_is_p2: "我们关注：关键低点或高点倾向于何时出现，用从股票上市日开始的周索引来衡量。",
        example_title: "示例：",
        section_what_is_example: "如果一只股票在第52、366、680、994周出现重要低点……我们寻找这些数字中的数学规律来预测下一个低点。",
        section_how_title: "工作原理：分步说明",
        step1_title: "收集历史数据",
        step1_desc: "从股票IPO到现在的每周OHLCV数据。",
        step2_title: "检测形态点",
        step2_desc: "使用ZigZag算法（5%阈值）识别重要的价格反转。",
        step3_title: "提取周索引",
        step3_desc: "将形态点日期转换为从IPO开始的周数（如第366周）。",
        step4_title: "寻找数学规律",
        step4_desc: "分析序列中的算术/几何规律（如：52、366、680...）。",
        step5_title: "计算下一个形态点",
        step5_desc: "应用规律预测下一次出现的周数。",
        step6_title: "转换为日期范围",
        step6_desc: "将周数转换为日历日期范围。",
        section_pattern_title: "规律示例：601933",
        table_col_landmark: "形态点",
        table_col_week: "周索引",
        table_col_gap: "间隔",
        table_col_pattern: "规律",
        pattern_base: "基准",
        pattern_2nd: "第2点",
        pattern_314: "314周间隔确认！",
        pattern_next: "下一个：680 + 314 = 994",
        pattern_note: "注：规律频率表示该间隔在历史上重复的次数。",
        section_week_title: "理解周索引",
        section_week_p1: "周索引计算从股票IPO（首次公开募股）开始的周数：",
        section_week_li1: "第0周 = 交易第一周（IPO周）",
        section_week_li2: "第52周 = IPO后约1年",
        section_week_li3: "第640周 = IPO后约12.3年",
        section_week_p2: "通过使用周索引而非日历日期，我们可以发现随时间重复的数学规律，无论市场状况、新闻事件或经济周期如何。",
        section_confidence_title: "置信度说明",
        confidence_high: "高",
        confidence_high_desc: "规律频率≥3。检测到强烈的重复规律。",
        section_disclaimer_title: "重要免责声明",
        disclaimer_not_financial_advice: "⚠️ 这不是投资建议",
        disclaimer_for_educational: "本工具仅供教育和研究目的",
        disclaimer_past_performance: "过去的规律不能保证未来的结果",
        disclaimer_do_research: "在做出投资决策前，务必自行研究",
        disclaimer_consult_professional: "咨询合格的专业理财顾问获取投资指导",
        section_faq_title: "常见问题",
        faq1_q: "什么是ZigZag算法？",
        faq1_a: "ZigZag算法过滤掉低于阈值（默认5%）的微小价格波动。它只识别重要的趋势反转，帮助我们专注于有意义的形态点而非噪音。",
        faq2_q: "为什么使用周数据而不是日数据？",
        faq2_a: "周数据平滑了每日噪音，减少了数据点数量，使规律更明显。形态点通常持续多天，因此周聚合保留了信号同时降低了噪音。",
        faq3_q: "这些预测有多准确？",
        faq3_a: "准确性在不同股票间差异很大，取决于规律强度。历史较长、规律更一致的股票往往有更高置信度的预测。我们正在追踪历史准确性以提供统计数据。",
        faq4_q: "我可以将其用于日内交易吗？",
        faq4_a: "不可以。本工具按周预测形态点，而非每日价格变动。它专为长期分析设计，而非短期交易。预测指示形态点可能出现的周范围，而非具体日期或价格。",
        cta_title: "准备好试试了吗？",
        cta_subtitle: "搜索任意A股股票查看其形态预测",
        cta_button: "开始分析股票",

        // History Page
        page_title_history: "历史记录 - 股票形态预测",
        history_title: "预测历史与验证",
        history_coming_soon_title: "功能即将推出",
        history_coming_soon_text: "历史预测跟踪和验证功能正在开发中。此页面将显示过去的预测、准确性以及预测表现的统计分析。",
        history_coming_soon_detail: "<strong>计划功能：</strong><br>• 所有历史预测的数据库<br>• 实际结果跟踪（形态点是否出现）<br>• 按置信度分类的准确性统计<br>• 形态表现分析<br>• 导出预测历史为CSV",
        history_table_title: "历史预测",
        history_no_data_title: "暂无历史数据",
        history_no_data_text: "预测跟踪将在功能实现后开始。请稍后查看更新！",
        hist_table_date_made: "预测日期",
        hist_table_symbol: "股票代码",
        hist_table_type: "类型",
        hist_table_predicted_week: "预测周",
        hist_table_predicted_range: "预测范围",
        hist_table_actual_week: "实际周",
        hist_table_result: "结果",
        hist_table_confidence: "置信度",
        stat_total_predictions: "总预测数",
        stat_accuracy: "总体准确率",
        stat_high_confidence_accuracy: "高置信度准确率"
    }
};

/**
 * Get current language (from localStorage or browser setting)
 */
function getCurrentLanguage() {
    const saved = localStorage.getItem('language');
    if (saved && (saved === 'en' || saved === 'zh')) {
        return saved;
    }
    // Detect from browser
    const browserLang = navigator.language || navigator.userLanguage;
    return browserLang.startsWith('zh') ? 'zh' : 'en';
}

/**
 * Set current language
 */
function setLanguage(lang) {
    localStorage.setItem('language', lang);
    location.reload();
}

/**
 * Get translation for a key
 */
function t(key) {
    const lang = getCurrentLanguage();
    return translations[lang][key] || translations['en'][key] || key;
}

/**
 * Apply translations to a page
 */
function applyTranslations() {
    const lang = getCurrentLanguage();

    // Set HTML lang attribute
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';

    // Translate elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        const translation = t(key);
        if (el.tagName === 'INPUT' && el.hasAttribute('placeholder')) {
            el.placeholder = translation;
        } else {
            el.textContent = translation;
        }
    });

    // Update language selector
    const selector = document.getElementById('languageSelector');
    if (selector) {
        selector.value = lang;
    }
}

// Initialize on DOM content loaded
document.addEventListener('DOMContentLoaded', () => {
    applyTranslations();
});
