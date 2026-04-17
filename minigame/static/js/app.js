/**
 * Food Annotation Game - Frontend
 * Supports both normal mode and blind mode
 */

const state = {
    currentDataset: null,
    currentDish: null,
    currentPlayer: null,  // Current annotator name
    gtVerified: {},
    aiVerified: {},
    equivalences: {},
    userAdded: [],
    challengeFlag: false,
    autoMatches: {},
    stats: {},
    // Blind mode state
    blindMode: false,
    columnAIsGt: true,  // In blind mode: which column shows GT
    colARejected: [],   // Items rejected from column A (blind mode)
    colBRejected: [],   // Items rejected from column B (blind mode)
    colAVerified: [],   // Items explicitly verified in column A (blind mode)
    colBVerified: [],   // Items explicitly verified in column B (blind mode)
    links: [],          // Bidirectional links [{a: item, b: item}] (blind mode)
    // Single-blind mode state
    singleBlindMode: false,
    phase: 1,           // Phase 1: merged list (per-image), Phase 2: global equivalency review
    currentSet: 'main', // 'main' or 'challenge' (images with unsure items)
    mergedIngredients: [],  // [{name, source: 'gt'|'ai'|'both', gt_name, ai_name, fuzzy_match}]
    approvedIngredients: [], // Ingredients approved in Phase 1
    rejectedIngredients: [], // Ingredients rejected in Phase 1
    unsureIngredients: [],   // Ingredients marked unsure (?) in Phase 1
    // Global Phase 2 state
    equivCandidates: [],     // [{term_a, term_b, count, images}]
    equivDecisions: {},      // {pairKey: {term_a, term_b, equivalent: bool}}
};

let isDragging = false;
let dragStartIngredient = null;
let dragStartColumn = null;

const el = {
    datasetSelect: document.getElementById('dataset-select'),
    currentImage: document.getElementById('current-image'),
    imageOriginalJaccard: document.getElementById('image-original-jaccard'),
    imageUpdatedJaccard: document.getElementById('image-updated-jaccard'),
    datasetOriginalJaccard: document.getElementById('dataset-original-jaccard'),
    datasetUpdatedJaccard: document.getElementById('dataset-updated-jaccard'),
    progressFill: document.getElementById('progress-fill'),
    progressText: document.getElementById('progress-text'),
    validatedText: document.getElementById('validated-text'),
    foodImage: document.getElementById('food-image'),
    imageContainer: document.getElementById('image-container'),
    magnifierLens: document.getElementById('magnifier-lens'),
    loadingOverlay: document.getElementById('loading-overlay'),
    aiDescriptionText: document.getElementById('ai-description-text'),
    aiDescriptionSection: document.getElementById('ai-description-section'),
    colABadge: document.getElementById('col-a-badge'),
    colBBadge: document.getElementById('col-b-badge'),
    colATitle: document.getElementById('col-a-title'),
    colBTitle: document.getElementById('col-b-title'),
    btnReset: document.getElementById('btn-reset'),
    btnSkip: document.getElementById('btn-skip'),
    btnSubmit: document.getElementById('btn-submit'),
    submitHint: document.getElementById('submit-hint'),
    colACards: document.getElementById('col-a-cards'),
    colBCards: document.getElementById('col-b-cards'),
    annotationPanel: document.getElementById('annotation-panel'),
    connectionLines: document.getElementById('connection-lines'),
    dragLineSvg: document.getElementById('drag-line-svg'),
    dragLine: document.getElementById('drag-line'),
    newIngredientInput: document.getElementById('new-ingredient-input'),
    btnAddIngredient: document.getElementById('btn-add-ingredient'),
    addIngredientSection: document.getElementById('add-ingredient-section'),
    newIngredientsSection: document.getElementById('new-ingredients-section'),
    blindNewIngredientInput: document.getElementById('blind-new-ingredient-input'),
    btnBlindAddIngredient: document.getElementById('btn-blind-add-ingredient'),
    newIngredientsList: document.getElementById('new-ingredients-list'),
    doneOverlay: document.getElementById('done-overlay'),
    finalJaccard: document.getElementById('final-jaccard'),
    // Single-blind mode elements
    singleBlindPhase1: document.getElementById('single-blind-phase1'),
    mergedCards: document.getElementById('merged-cards'),
    singleBlindNewInput: document.getElementById('single-blind-new-input'),
    btnSingleBlindAdd: document.getElementById('btn-single-blind-add'),
    // Global Phase 2 elements
    globalPhase2: document.getElementById('global-phase2'),
    phase2CandidatesList: document.getElementById('phase2-candidates-list'),
    phase2ProgressText: document.getElementById('phase2-progress-text'),
    phase2ProgressFill: document.getElementById('phase2-progress-fill'),
    btnPhase2Finish: document.getElementById('btn-phase2-finish'),
    btnProceedPhase2: document.getElementById('btn-proceed-phase2'),
    btnDoneReload: document.getElementById('btn-done-reload'),
    doneTitle: document.getElementById('done-title'),
    doneMessage: document.getElementById('done-message'),
    // Set toggle elements
    setToggleItem: document.getElementById('set-toggle-item'),
    btnMainSet: document.getElementById('btn-main-set'),
    btnChallengeSet: document.getElementById('btn-challenge-set'),
    mainSetCount: document.getElementById('main-set-count'),
    challengeSetCount: document.getElementById('challenge-set-count'),
    // Dashboard elements
    btnDashboard: document.getElementById('btn-dashboard'),
    btnExport: document.getElementById('btn-export'),
    dashboardPanel: document.getElementById('dashboard-panel'),
    btnDashboardBack: document.getElementById('btn-dashboard-back'),
    mainContent: document.querySelector('.main-content'),
    // Dashboard stat elements
    dashCompletedCount: document.getElementById('dash-completed-count'),
    dashCompletedDetail: document.getElementById('dash-completed-detail'),
    dashProgressPct: document.getElementById('dash-progress-pct'),
    dashProgressDetail: document.getElementById('dash-progress-detail'),
    dashGtPrecisionStrict: document.getElementById('dash-gt-precision-strict'),
    dashGtPrecisionUncertain: document.getElementById('dash-gt-precision-uncertain'),
    dashGtApproved: document.getElementById('dash-gt-approved'),
    dashGtRejected: document.getElementById('dash-gt-rejected'),
    dashGtUnsure: document.getElementById('dash-gt-unsure'),
    dashAiPrecisionStrict: document.getElementById('dash-ai-precision-strict'),
    dashAiPrecisionUncertain: document.getElementById('dash-ai-precision-uncertain'),
    dashAiApproved: document.getElementById('dash-ai-approved'),
    dashAiRejected: document.getElementById('dash-ai-rejected'),
    dashAiUnsure: document.getElementById('dash-ai-unsure'),
    dashCutoffDate: document.getElementById('dash-cutoff-date'),
    precisionChart: document.getElementById('precision-chart'),
    breakdownChart: document.getElementById('breakdown-chart'),
    excludePerfectToggle: document.getElementById('exclude-perfect-toggle'),
    annotatorFilterSelect: document.getElementById('annotator-filter-select'),
    // Player selection elements
    playerModal: document.getElementById('player-modal'),
    playerButtonsContainer: document.getElementById('player-buttons'),
    customPlayerInput: document.getElementById('custom-player-input'),
    btnCustomPlayer: document.getElementById('btn-custom-player'),
    currentPlayerName: document.getElementById('current-player-name'),
    playerNameBtn: document.getElementById('player-name-btn')
};

// ============================================
// Player Selection
// ============================================

async function showPlayerModal() {
    if (!el.playerModal) return;
    
    // Populate player buttons dynamically
    if (el.playerButtonsContainer && state.currentDataset) {
        try {
            const annotators = await fetch(`api/annotators/${state.currentDataset}`).then(r => r.json());
            
            // Clear existing buttons
            el.playerButtonsContainer.innerHTML = '';
            
            // Add button for each annotator
            annotators.forEach(name => {
                const btn = document.createElement('button');
                btn.className = 'player-btn';
                btn.textContent = name;
                btn.addEventListener('click', () => setPlayer(name));
                el.playerButtonsContainer.appendChild(btn);
            });
            
            // If no annotators yet, show a message
            if (annotators.length === 0) {
                el.playerButtonsContainer.innerHTML = '<span class="no-annotators">No previous annotators. Enter your name below.</span>';
            }
        } catch (err) {
            console.error('Failed to load annotators:', err);
            el.playerButtonsContainer.innerHTML = '<span class="no-annotators">Enter your name below.</span>';
        }
    }
    
    el.playerModal.classList.add('visible');
}

function hidePlayerModal() {
    if (el.playerModal) el.playerModal.classList.remove('visible');
}

function setPlayer(name) {
    if (!name || !name.trim()) return;
    
    state.currentPlayer = name.trim();
    localStorage.setItem('annotator', state.currentPlayer);
    
    // Update display
    if (el.currentPlayerName) {
        el.currentPlayerName.textContent = state.currentPlayer;
    }
    
    hidePlayerModal();
}

function loadSavedPlayer() {
    const saved = localStorage.getItem('annotator');
    if (saved) {
        state.currentPlayer = saved;
        if (el.currentPlayerName) {
            el.currentPlayerName.textContent = saved;
        }
        return true;
    }
    return false;
}

function setupPlayerSelection() {
    // Custom player input
    if (el.btnCustomPlayer) {
        el.btnCustomPlayer.addEventListener('click', () => {
            setPlayer(el.customPlayerInput.value);
        });
    }
    if (el.customPlayerInput) {
        el.customPlayerInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') setPlayer(el.customPlayerInput.value);
        });
    }
    
    // Click player name to change
    if (el.playerNameBtn) {
        el.playerNameBtn.addEventListener('click', showPlayerModal);
    }
}

// ============================================
// Magnifier
// ============================================

const MAGNIFIER_ZOOM = 1.5;  // Zoom level

function setupMagnifier() {
    if (!el.imageContainer || !el.magnifierLens || !el.foodImage) return;
    
    el.imageContainer.addEventListener('mouseenter', showMagnifier);
    el.imageContainer.addEventListener('mouseleave', hideMagnifier);
    el.imageContainer.addEventListener('mousemove', moveMagnifier);
}

function showMagnifier() {
    if (!el.foodImage.src || el.foodImage.src === window.location.href) return;
    el.magnifierLens.classList.add('active');
}

function hideMagnifier() {
    el.magnifierLens.classList.remove('active');
}

function moveMagnifier(e) {
    if (!el.magnifierLens.classList.contains('active')) return;
    
    const container = el.imageContainer;
    const img = el.foodImage;
    const lens = el.magnifierLens;
    
    // Get container and image dimensions
    const containerRect = container.getBoundingClientRect();
    const imgRect = img.getBoundingClientRect();
    
    // Mouse position relative to container
    const mouseX = e.clientX - containerRect.left;
    const mouseY = e.clientY - containerRect.top;
    
    // Position lens centered on cursor
    lens.style.left = mouseX + 'px';
    lens.style.top = mouseY + 'px';
    
    // Calculate position relative to the image (not container)
    const imgOffsetX = imgRect.left - containerRect.left;
    const imgOffsetY = imgRect.top - containerRect.top;
    
    // Mouse position relative to image
    const imgMouseX = e.clientX - imgRect.left;
    const imgMouseY = e.clientY - imgRect.top;
    
    // Check if mouse is over the actual image
    if (imgMouseX < 0 || imgMouseX > imgRect.width || 
        imgMouseY < 0 || imgMouseY > imgRect.height) {
        lens.style.backgroundImage = 'none';
        return;
    }
    
    // Set background image and calculate position
    const lensWidth = lens.offsetWidth;
    const lensHeight = lens.offsetHeight;
    
    // Background size is image size * zoom
    const bgWidth = imgRect.width * MAGNIFIER_ZOOM;
    const bgHeight = imgRect.height * MAGNIFIER_ZOOM;
    
    // Background position: center the zoomed area on the cursor
    const bgX = -(imgMouseX * MAGNIFIER_ZOOM - lensWidth / 2);
    const bgY = -(imgMouseY * MAGNIFIER_ZOOM - lensHeight / 2);
    
    lens.style.backgroundImage = `url('${img.src}')`;
    lens.style.backgroundSize = `${bgWidth}px ${bgHeight}px`;
    lens.style.backgroundPosition = `${bgX}px ${bgY}px`;
}

// API
async function fetchConfig() { return (await fetch('/api/config')).json(); }
async function fetchDatasets() { return (await fetch('/api/datasets')).json(); }
async function fetchStats(dataset) { return (await fetch(`/api/stats/${dataset}`)).json(); }
async function fetchNext(dataset, imageSet = 'main') { 
    return (await fetch(`/api/next/${dataset}?set=${imageSet}`)).json(); 
}
async function submitAnnotation(data) {
    return (await fetch('/api/annotate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })).json();
}
async function fetchEquivCandidates(dataset) {
    return (await fetch(`/api/equivalency-candidates/${dataset}`)).json();
}
async function submitGlobalEquivalencies(data) {
    return (await fetch('/api/save-equivalencies', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })).json();
}

// Fuzzy matching (mirrors backend)
const MODIFIERS = new Set(['roasted','grilled','fried','baked','steamed','sauteed','boiled','raw','fresh','frozen','canned','dried','cooked','sliced','diced','chopped','minced','cubed','shredded','red','green','yellow','orange','white','black','brown','purple','sweet','sour','spicy','large','small','medium','organic','natural','plain','seasoned','baby','young','smoked','cured','pickled','mashed','pureed','crushed','ground','crispy','crunchy','soft','tender','mixed','assorted','various']);

function normalize(name) { return name.toLowerCase().trim().replace(/\s+/g, ' '); }
function getBase(name) {
    let words = normalize(name).split(' ');
    while (words.length && MODIFIERS.has(words[0])) words.shift();
    while (words.length && MODIFIERS.has(words[words.length-1])) words.pop();
    return words.join(' ') || normalize(name);
}
function fuzzyMatch(a, b) {
    const n1 = normalize(a), n2 = normalize(b);
    if (n1 === n2) return true;
    const b1 = getBase(a), b2 = getBase(b);
    if (b1 === b2 || b1.includes(b2) || b2.includes(b1)) return true;
    if (b1.replace(/s$/, '') === b2.replace(/s$/, '')) return true;
    return false;
}

// UI
function showLoading(show) { el.loadingOverlay.classList.toggle('visible', show); }
function showDone(jaccard) {
    el.doneOverlay.classList.add('visible');
    el.finalJaccard.textContent = (jaccard ?? 0).toFixed(4);
    
    // In single-blind mode, Phase 1 finishes all images first, then user can optionally do global Phase 2
    if (state.singleBlindMode) {
        if (el.doneTitle) el.doneTitle.textContent = 'Phase 1 Complete';
        if (el.doneMessage) el.doneMessage.textContent = 'You reviewed all images. Next: optional equivalency review aggregated across the whole dataset.';
        if (el.btnProceedPhase2) el.btnProceedPhase2.style.display = 'inline-flex';
        if (el.btnDoneReload) el.btnDoneReload.style.display = 'none';
    } else {
        if (el.doneTitle) el.doneTitle.textContent = 'All Done!';
        if (el.doneMessage) el.doneMessage.textContent = "You've annotated all available images in this dataset.";
        if (el.btnProceedPhase2) el.btnProceedPhase2.style.display = 'none';
        if (el.btnDoneReload) el.btnDoneReload.style.display = 'inline-flex';
    }
}

function showDoneWithChallenge() {
    // Main set done, but there are challenge items to review
    el.doneOverlay.classList.add('visible');
    el.finalJaccard.textContent = (state.stats.updated_jaccard_mean ?? 0).toFixed(4);
    if (el.doneTitle) el.doneTitle.textContent = 'Main Set Complete!';
    if (el.doneMessage) el.doneMessage.innerHTML = `
        You've completed the main set.<br>
        <strong>${state.stats.challenge_count}</strong> images have "unsure" items in the Challenge set.<br>
        You can review them now or proceed to equivalency review.
    `;
    if (el.btnProceedPhase2) el.btnProceedPhase2.style.display = 'inline-flex';
    if (el.btnDoneReload) el.btnDoneReload.textContent = 'Go to Challenge Set';
    if (el.btnDoneReload) {
        el.btnDoneReload.style.display = 'inline-flex';
        el.btnDoneReload.onclick = () => {
            el.doneOverlay.classList.remove('visible');
            switchToSet('challenge');
        };
    }
}

function switchToSet(setName) {
    state.currentSet = setName;
    
    // Update toggle buttons
    if (el.btnMainSet) el.btnMainSet.classList.toggle('active', setName === 'main');
    if (el.btnChallengeSet) el.btnChallengeSet.classList.toggle('active', setName === 'challenge');
    
    // Load next image from the selected set
    loadNextImage();
}

function updateStats(stats) {
    state.stats = stats;
    el.datasetOriginalJaccard.textContent = stats.original_jaccard_mean.toFixed(3);
    el.datasetUpdatedJaccard.textContent = stats.updated_jaccard_mean.toFixed(3);
    
    // Progress bar shows annotated/(total - perfect) to focus on manual work needed
    const needsAnnotation = stats.total_images - stats.perfect_jaccard_count;
    const pct = needsAnnotation > 0 ? stats.annotated_count / needsAnnotation : 1;
    el.progressFill.style.width = `${pct * 100}%`;
    el.progressText.textContent = `${stats.annotated_count}/${needsAnnotation}`;
    
    // Show breakdown below
    el.validatedText.textContent = `(${stats.perfect_jaccard_count} perfect + ${stats.annotated_count} annotated)`;
    
    // Update set toggle counts (single-blind mode)
    if (el.mainSetCount) el.mainSetCount.textContent = stats.remaining || 0;
    if (el.challengeSetCount) el.challengeSetCount.textContent = stats.challenge_count || 0;
    
    // Show set toggle if there are challenge items
    if (el.setToggleItem && state.singleBlindMode) {
        el.setToggleItem.style.display = (stats.challenge_count > 0) ? 'flex' : 'none';
    }
}

function resetState() {
    state.gtVerified = {};
    state.aiVerified = {};
    state.equivalences = {};
    state.userAdded = [];
    state.challengeFlag = false;
    state.autoMatches = {};
    // Blind mode state
    state.colARejected = [];
    state.colBRejected = [];
    state.colAVerified = [];
    state.colBVerified = [];
    state.links = [];
    // Single-blind mode state
    state.phase = 1;
    state.mergedIngredients = [];
    state.approvedIngredients = [];
    state.rejectedIngredients = [];
    state.unsureIngredients = [];
    
    el.connectionLines.innerHTML = '';
}

function getCardStatus(ingredient, type) {
    if (state.blindMode) {
        // Blind mode: check column-based rejection, verification, and linking
        const col = type; // 'a' or 'b'
        const rejected = col === 'a' ? state.colARejected : state.colBRejected;
        const verified = col === 'a' ? state.colAVerified : state.colBVerified;
        
        if (rejected.includes(ingredient)) return 'rejected';
        if (verified.includes(ingredient)) return 'verified';
        
        // Check if linked (manual link)
        const isLinked = state.links.some(l => l.a === ingredient || l.b === ingredient);
        if (isLinked) return 'linked';
        
        // Check auto-match (fuzzy match between columns)
        const otherCol = col === 'a' ? getColBItems() : getColAItems();
        const otherRejected = col === 'a' ? state.colBRejected : state.colARejected;
        const hasAutoMatch = otherCol.some(other => 
            !otherRejected.includes(other) && fuzzyMatch(ingredient, other)
        );
        if (hasAutoMatch) return 'linked';
        
        return 'unlinked';
    }
    
    // Normal mode
    if (type === 'gt') return state.gtVerified[ingredient] === 'rejected' ? 'rejected' : 'verified';
    const status = state.aiVerified[ingredient];
    if (status === 'verified') return 'verified';  // User confirmed AI is correct (turns green)
    if (status === 'rejected') return 'rejected';
    if (state.equivalences[ingredient]?.length > 0) return 'equivalent';
    if (state.autoMatches[ingredient]) return 'matched';
    return 'unmatched';
}

// Helper functions for blind mode
function getColAItems() {
    if (!state.currentDish) return [];
    return state.columnAIsGt ? state.currentDish.gt_ingredients : state.currentDish.ai_ingredients;
}

function getColBItems() {
    if (!state.currentDish) return [];
    return state.columnAIsGt ? state.currentDish.ai_ingredients : state.currentDish.gt_ingredients;
}

function alignEquivalentItems(colAItems, colBItems) {
    // Align items so that AUTO fuzzy-matched pairs appear on the same row
    // Manual links and unmatched items don't need alignment
    const alignedA = [];
    const alignedB = [];
    const usedA = new Set();
    const usedB = new Set();
    
    // First pass: find AUTO fuzzy matches only (not manual links)
    for (const aItem of colAItems) {
        if (state.colARejected.includes(aItem)) continue;
        
        // Only align auto fuzzy matches
        for (const bItem of colBItems) {
            if (usedB.has(bItem) || state.colBRejected.includes(bItem)) continue;
            if (fuzzyMatch(aItem, bItem)) {
                alignedA.push(aItem);
                alignedB.push(bItem);
                usedA.add(aItem);
                usedB.add(bItem);
                break;
            }
        }
    }
    
    // Second pass: add remaining items from both columns in parallel (same rows, no placeholders)
    const remainingA = colAItems.filter(i => !usedA.has(i) && !state.colARejected.includes(i));
    const remainingB = colBItems.filter(i => !usedB.has(i) && !state.colBRejected.includes(i));
    const maxRemaining = Math.max(remainingA.length, remainingB.length);
    
    for (let i = 0; i < maxRemaining; i++) {
        alignedA.push(remainingA[i] || null);
        alignedB.push(remainingB[i] || null);
    }
    
    // Third pass: add rejected items at the end
    const rejectedA = colAItems.filter(i => state.colARejected.includes(i));
    const rejectedB = colBItems.filter(i => state.colBRejected.includes(i));
    const maxRejected = Math.max(rejectedA.length, rejectedB.length);
    
    for (let i = 0; i < maxRejected; i++) {
        alignedA.push(rejectedA[i] || null);
        alignedB.push(rejectedB[i] || null);
    }
    
    return { alignedA, alignedB };
}

function createBlindCard(ingredient, column) {
    // Blind mode card - same for both columns
    const card = document.createElement('div');
    card.className = 'ingredient-card';
    card.dataset.ingredient = ingredient;
    card.dataset.column = column;
    const status = getCardStatus(ingredient, column);
    card.classList.add(status);

    // Drag handle for linking
    const handle = document.createElement('div');
    handle.className = 'drag-handle' + (status === 'linked' ? ' linked' : '');
    handle.onmousedown = (e) => startDrag(e, ingredient, column);
    card.appendChild(handle);

    const name = document.createElement('span');
    name.className = 'ingredient-name';
    name.textContent = ingredient;
    card.appendChild(name);

    const actions = document.createElement('div');
    actions.className = 'card-actions';

    // Check if this item is verified (not rejected and either linked or explicitly verified)
    const rejected = column === 'a' ? state.colARejected : state.colBRejected;
    const verified = column === 'a' ? state.colAVerified : state.colBVerified;
    const isRejected = rejected.includes(ingredient);
    const isVerified = verified?.includes(ingredient) || status === 'linked';

    // Verify button
    const verifyBtn = document.createElement('button');
    verifyBtn.className = 'card-btn btn-verify' + (isVerified && !isRejected ? ' active' : '');
    verifyBtn.innerHTML = '✓';
    verifyBtn.onclick = (e) => { e.stopPropagation(); handleBlindVerify(ingredient, column); };
    actions.appendChild(verifyBtn);

    // Reject button
    const rejectBtn = document.createElement('button');
    rejectBtn.className = 'card-btn btn-reject' + (isRejected ? ' active' : '');
    rejectBtn.innerHTML = '✕';
    rejectBtn.onclick = (e) => { e.stopPropagation(); handleBlindReject(ingredient, column); };
    actions.appendChild(rejectBtn);

    card.appendChild(actions);
    return card;
}

function createGtCard(ingredient, isUserAdded = false) {
    const card = document.createElement('div');
    card.className = 'ingredient-card';
    card.dataset.ingredient = ingredient;
    card.dataset.type = 'gt';
    const status = getCardStatus(ingredient, 'gt');
    card.classList.add(status);
    if (isUserAdded) card.classList.add('user-added');

    const name = document.createElement('span');
    name.className = 'ingredient-name';
    name.textContent = ingredient;
    card.appendChild(name);

    const actions = document.createElement('div');
    actions.className = 'card-actions';

    const verifyBtn = document.createElement('button');
    verifyBtn.className = 'card-btn btn-verify' + (status === 'verified' ? ' active' : '');
    verifyBtn.innerHTML = '✓';
    verifyBtn.onclick = (e) => { e.stopPropagation(); handleVerify(ingredient, 'gt'); };
    actions.appendChild(verifyBtn);

    const rejectBtn = document.createElement('button');
    rejectBtn.className = 'card-btn btn-reject' + (status === 'rejected' ? ' active' : '');
    rejectBtn.innerHTML = '✕';
    rejectBtn.onclick = (e) => { e.stopPropagation(); handleReject(ingredient, 'gt', isUserAdded); };
    actions.appendChild(rejectBtn);

    card.appendChild(actions);
    return card;
}

function createAiCard(ingredient) {
    const card = document.createElement('div');
    card.className = 'ingredient-card';
    card.dataset.ingredient = ingredient;
    card.dataset.type = 'ai';
    const status = getCardStatus(ingredient, 'ai');
    card.classList.add(status);

    const handle = document.createElement('div');
    handle.className = 'drag-handle' + (status === 'matched' || status === 'equivalent' ? ' linked' : '');
    handle.onmousedown = (e) => startDrag(e, ingredient, 'ai');
    card.appendChild(handle);

    const name = document.createElement('span');
    name.className = 'ingredient-name';
    name.textContent = ingredient;
    card.appendChild(name);

    const actions = document.createElement('div');
    actions.className = 'card-actions';

    const verifyBtn = document.createElement('button');
    verifyBtn.className = 'card-btn btn-verify' + (status === 'verified' ? ' active' : '');
    verifyBtn.innerHTML = '✓';
    verifyBtn.onclick = (e) => { e.stopPropagation(); handleVerify(ingredient, 'ai'); };
    actions.appendChild(verifyBtn);

    const rejectBtn = document.createElement('button');
    rejectBtn.className = 'card-btn btn-reject' + (status === 'rejected' ? ' active' : '');
    rejectBtn.innerHTML = '✕';
    rejectBtn.onclick = (e) => { e.stopPropagation(); handleReject(ingredient, 'ai'); };
    actions.appendChild(rejectBtn);

    card.appendChild(actions);
    return card;
}

function handleBlindVerify(ingredient, column) {
    const verified = column === 'a' ? state.colAVerified : state.colBVerified;
    const rejected = column === 'a' ? state.colARejected : state.colBRejected;
    
    // Remove from rejected if present
    const rejIdx = rejected.indexOf(ingredient);
    if (rejIdx >= 0) rejected.splice(rejIdx, 1);
    
    // Toggle in verified
    const verIdx = verified.indexOf(ingredient);
    if (verIdx >= 0) {
        verified.splice(verIdx, 1); // Un-verify
    } else {
        verified.push(ingredient);
    }
    
    renderCards();
    updateLocalJaccard();
    updateSubmitButton();
}

function handleBlindReject(ingredient, column) {
    const rejected = column === 'a' ? state.colARejected : state.colBRejected;
    const verified = column === 'a' ? state.colAVerified : state.colBVerified;
    
    // Remove from verified if present
    const verIdx = verified.indexOf(ingredient);
    if (verIdx >= 0) verified.splice(verIdx, 1);
    
    const idx = rejected.indexOf(ingredient);
    if (idx >= 0) {
        rejected.splice(idx, 1); // Toggle off
    } else {
        rejected.push(ingredient); // Toggle on
        // Remove any links involving this ingredient
        state.links = state.links.filter(l => l.a !== ingredient && l.b !== ingredient);
    }
    renderCards();
    updateLocalJaccard();
    updateSubmitButton();
}

// ============================================================================
// Single-Blind Mode Functions
// ============================================================================

function createMergedCard(item, isUserAdded = false) {
    // Phase 1 card for merged ingredient list
    const card = document.createElement('div');
    card.className = 'ingredient-card merged-card';
    card.dataset.ingredient = item.name;
    card.dataset.source = item.source;
    if (isUserAdded) card.dataset.userAdded = 'true';
    
    // Determine card status
    const isApproved = state.approvedIngredients.includes(item.name);
    const isRejected = state.rejectedIngredients.includes(item.name);
    const isUnsure = state.unsureIngredients.includes(item.name);
    
    if (isApproved) card.classList.add('approved');
    else if (isRejected) card.classList.add('rejected');
    else if (isUnsure) card.classList.add('unsure');
    else card.classList.add('pending');
    
    // Source indicator (hidden from user in blind mode, but useful for debugging)
    const sourceIndicator = document.createElement('span');
    sourceIndicator.className = 'source-indicator';
    sourceIndicator.dataset.source = item.source;
    card.appendChild(sourceIndicator);
    
    const name = document.createElement('span');
    name.className = 'ingredient-name';
    name.textContent = item.name;
    card.appendChild(name);
    
    const actions = document.createElement('div');
    actions.className = 'card-actions';
    
    // Approve button (✓ = present)
    const approveBtn = document.createElement('button');
    approveBtn.className = 'card-btn btn-verify' + (isApproved ? ' active' : '');
    approveBtn.innerHTML = '✓';
    approveBtn.title = 'Present in image';
    approveBtn.onclick = (e) => { e.stopPropagation(); handleSingleBlindApprove(item.name); };
    actions.appendChild(approveBtn);
    
    // Reject button (✕ = not present)
    const rejectBtn = document.createElement('button');
    rejectBtn.className = 'card-btn btn-reject' + (isRejected ? ' active' : '');
    rejectBtn.innerHTML = '✕';
    rejectBtn.title = 'Not present in image';
    rejectBtn.onclick = (e) => { e.stopPropagation(); handleSingleBlindReject(item.name, isUserAdded); };
    actions.appendChild(rejectBtn);
    
    // Unsure button (? = not sure)
    const unsureBtn = document.createElement('button');
    unsureBtn.className = 'card-btn btn-unsure' + (isUnsure ? ' active' : '');
    unsureBtn.innerHTML = '?';
    unsureBtn.title = 'Unsure';
    unsureBtn.onclick = (e) => { e.stopPropagation(); handleSingleBlindUnsure(item.name); };
    actions.appendChild(unsureBtn);
    
    card.appendChild(actions);
    return card;
}

function handleSingleBlindApprove(ingredient) {
    // Remove from rejected and unsure if present
    state.rejectedIngredients = state.rejectedIngredients.filter(i => i !== ingredient);
    state.unsureIngredients = state.unsureIngredients.filter(i => i !== ingredient);
    
    // Toggle in approved
    const appIdx = state.approvedIngredients.indexOf(ingredient);
    if (appIdx >= 0) {
        state.approvedIngredients.splice(appIdx, 1); // Un-approve
    } else {
        state.approvedIngredients.push(ingredient);
    }
    
    renderSingleBlindPhase1();
    updateSingleBlindSubmitButton();
}

function handleSingleBlindReject(ingredient, isUserAdded = false) {
    // Remove from approved and unsure if present
    state.approvedIngredients = state.approvedIngredients.filter(i => i !== ingredient);
    state.unsureIngredients = state.unsureIngredients.filter(i => i !== ingredient);
    
    if (isUserAdded) {
        // Remove user-added ingredient entirely
        state.userAdded = state.userAdded.filter(i => i !== ingredient);
        state.mergedIngredients = state.mergedIngredients.filter(i => i.name !== ingredient);
    } else {
        // Toggle in rejected
        const rejIdx = state.rejectedIngredients.indexOf(ingredient);
        if (rejIdx >= 0) {
            state.rejectedIngredients.splice(rejIdx, 1); // Un-reject
        } else {
            state.rejectedIngredients.push(ingredient);
        }
    }
    
    renderSingleBlindPhase1();
    updateSingleBlindSubmitButton();
}

function handleSingleBlindUnsure(ingredient) {
    // Remove from approved and rejected if present
    state.approvedIngredients = state.approvedIngredients.filter(i => i !== ingredient);
    state.rejectedIngredients = state.rejectedIngredients.filter(i => i !== ingredient);
    
    // Toggle in unsure
    const unsIdx = state.unsureIngredients.indexOf(ingredient);
    if (unsIdx >= 0) {
        state.unsureIngredients.splice(unsIdx, 1); // Un-unsure
    } else {
        state.unsureIngredients.push(ingredient);
    }
    
    renderSingleBlindPhase1();
    updateSingleBlindSubmitButton();
}

function handleSingleBlindAddIngredient() {
    const input = el.singleBlindNewInput;
    const val = input.value.trim();
    if (val && !state.userAdded.includes(val)) {
        state.userAdded.push(val);
        // Add to merged list
        state.mergedIngredients.push({
            name: val,
            gt_name: null,
            ai_name: null,
            source: 'user',
            exact_match: false
        });
        // Auto-approve user-added ingredients
        state.approvedIngredients.push(val);
        input.value = '';
        renderSingleBlindPhase1();
        updateSingleBlindSubmitButton();
    }
}

function renderSingleBlindPhase1() {
    if (!el.mergedCards) return;
    el.mergedCards.innerHTML = '';
    
    // Render all items in their original order (don't reorder by status)
    // This keeps the list stable when users click buttons
    state.mergedIngredients.forEach(item => {
        el.mergedCards.appendChild(createMergedCard(item, item.source === 'user'));
    });
}

// ============================================================================
// Global Phase 2: Equivalency Review (after all images are done)
// ============================================================================

async function loadGlobalPhase2() {
    showLoading(true);
    
    try {
        const data = await fetchEquivCandidates(state.currentDataset);
        state.equivCandidates = data.candidates || [];
        state.equivDecisions = {};
        
        // Hide everything else, show Phase 2 panel
        document.querySelector('.main-content').style.display = 'none';
        el.doneOverlay.classList.remove('visible');
        if (el.globalPhase2) el.globalPhase2.style.display = 'flex';
        
        renderGlobalPhase2();
        updatePhase2Progress();
    } catch (err) {
        console.error('Failed to load equivalency candidates:', err);
    } finally {
        showLoading(false);
    }
}

function renderGlobalPhase2() {
    if (!el.phase2CandidatesList) return;
    el.phase2CandidatesList.innerHTML = '';
    
    if (state.equivCandidates.length === 0) {
        el.phase2CandidatesList.innerHTML = `
            <div class="phase2-no-candidates">
                <span class="done-icon">✨</span>
                <h3>No equivalency candidates found</h3>
                <p>All approved ingredients were automatically matched via fuzzy matching.<br>
                No manual equivalency review is needed.</p>
            </div>
        `;
        if (el.btnPhase2Finish) el.btnPhase2Finish.disabled = false;
        return;
    }
    
    state.equivCandidates.forEach((candidate, idx) => {
        const pairKey = `${candidate.term_a}|||${candidate.term_b}`;
        const decision = state.equivDecisions[pairKey];
        
        const card = document.createElement('div');
        card.className = 'equiv-candidate';
        if (decision?.equivalent === true) card.classList.add('decided-same');
        else if (decision?.equivalent === false) card.classList.add('decided-diff');
        card.dataset.pairKey = pairKey;
        
        // Terms display
        const terms = document.createElement('div');
        terms.className = 'equiv-terms';
        
        const termA = document.createElement('span');
        termA.className = 'equiv-term';
        termA.textContent = candidate.term_a;
        terms.appendChild(termA);
        
        const arrow = document.createElement('span');
        arrow.className = 'equiv-arrow';
        arrow.textContent = '↔';
        terms.appendChild(arrow);
        
        const termB = document.createElement('span');
        termB.className = 'equiv-term';
        termB.textContent = candidate.term_b;
        terms.appendChild(termB);
        
        card.appendChild(terms);
        
        // Count badge
        const count = document.createElement('span');
        count.className = 'equiv-count';
        count.textContent = `${candidate.count} image${candidate.count > 1 ? 's' : ''}`;
        card.appendChild(count);
        
        // Action buttons
        const actions = document.createElement('div');
        actions.className = 'equiv-actions';
        
        const sameBtn = document.createElement('button');
        sameBtn.className = 'card-btn btn-verify' + (decision?.equivalent === true ? ' active' : '');
        sameBtn.innerHTML = '✓';
        sameBtn.title = 'Same ingredient (equivalent)';
        sameBtn.onclick = () => handleEquivDecision(pairKey, candidate, true);
        actions.appendChild(sameBtn);
        
        const diffBtn = document.createElement('button');
        diffBtn.className = 'card-btn btn-reject' + (decision?.equivalent === false ? ' active' : '');
        diffBtn.innerHTML = '✕';
        diffBtn.title = 'Different ingredients';
        diffBtn.onclick = () => handleEquivDecision(pairKey, candidate, false);
        actions.appendChild(diffBtn);
        
        card.appendChild(actions);
        el.phase2CandidatesList.appendChild(card);
    });
}

function handleEquivDecision(pairKey, candidate, equivalent) {
    // Toggle: if same decision already set, un-decide
    if (state.equivDecisions[pairKey]?.equivalent === equivalent) {
        delete state.equivDecisions[pairKey];
    } else {
        state.equivDecisions[pairKey] = {
            term_a: candidate.term_a,
            term_b: candidate.term_b,
            equivalent: equivalent
        };
    }
    
    renderGlobalPhase2();
    updatePhase2Progress();
}

function updatePhase2Progress() {
    const total = state.equivCandidates.length;
    const decided = Object.keys(state.equivDecisions).length;
    
    if (el.phase2ProgressText) {
        el.phase2ProgressText.textContent = `${decided} / ${total} reviewed`;
    }
    if (el.phase2ProgressFill) {
        el.phase2ProgressFill.style.width = total > 0 ? `${(decided / total) * 100}%` : '0%';
    }
    if (el.btnPhase2Finish) {
        // Allow finishing any time; only ✓ decisions are saved as equivalences
        el.btnPhase2Finish.disabled = false;
    }
}

async function handlePhase2Finish() {
    showLoading(true);
    
    try {
        const decisions = Object.values(state.equivDecisions);
        
        const result = await submitGlobalEquivalencies({
            dataset: state.currentDataset,
            decisions: decisions
        });
        
        console.log('Equivalencies saved:', result);
        
        // Show completion
        if (el.globalPhase2) el.globalPhase2.style.display = 'none';
        
        // Show done overlay with final stats
        const stats = await fetchStats(state.currentDataset);
        updateStats(stats);
        
        el.doneTitle.textContent = 'All Done!';
        el.doneMessage.innerHTML = `
            Phase 1 complete: All images reviewed.<br>
            Phase 2 complete: ${result.equivalencies_saved || 0} equivalencies confirmed.<br>
            Jaccards recalculated for ${result.updated_count || 0} images.
        `;
        el.finalJaccard.textContent = stats.updated_jaccard_mean.toFixed(4);
        if (el.btnProceedPhase2) el.btnProceedPhase2.style.display = 'none';
        if (el.btnDoneReload) el.btnDoneReload.style.display = 'inline-flex';
        el.doneOverlay.classList.add('visible');
    } catch (err) {
        console.error('Failed to save equivalencies:', err);
    } finally {
        showLoading(false);
    }
}

function updateSingleBlindSubmitButton() {
    if (!state.singleBlindMode) return;
    
    // Phase 1 (per-image): All items must be marked as ✓ / ✕ / ?
    const hasPending = state.mergedIngredients.some(item =>
        !state.approvedIngredients.includes(item.name) &&
        !state.rejectedIngredients.includes(item.name) &&
        !state.unsureIngredients.includes(item.name)
    );
    el.btnSubmit.disabled = hasPending;
    el.submitHint.classList.toggle('visible', hasPending);
    el.submitHint.textContent = hasPending
        ? 'Mark all ingredients as present (✓), absent (✕), or unsure (?) to submit'
        : '';
}

function setupSingleBlindMode() {
    document.body.classList.add('single-blind-mode');
    
    // Hide AI description in single-blind mode (would give away which is AI)
    if (el.aiDescriptionSection) el.aiDescriptionSection.style.display = 'none';
    
    // Hide normal annotation panel, show single-blind Phase 1
    if (el.annotationPanel) el.annotationPanel.style.display = 'none';
    if (el.singleBlindPhase1) el.singleBlindPhase1.style.display = 'flex';
    if (el.globalPhase2) el.globalPhase2.style.display = 'none';
    
    // Submit button text for per-image Phase 1 flow
    const submitTextEl = el.btnSubmit?.querySelector('span:last-child');
    if (submitTextEl) submitTextEl.textContent = 'Submit & Next';
    
    // Hide Jaccard displays
    document.querySelectorAll('.jaccard-compare').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.stat-label').forEach(el => {
        if (el.textContent.includes('Jaccard') || el.textContent.includes('This Image')) {
            el.parentElement.style.display = 'none';
        }
    });
    
    // Setup event handlers
    if (el.btnSingleBlindAdd) {
        el.btnSingleBlindAdd.onclick = handleSingleBlindAddIngredient;
    }
    if (el.singleBlindNewInput) {
        el.singleBlindNewInput.onkeypress = (e) => { if (e.key === 'Enter') handleSingleBlindAddIngredient(); };
    }
}

function renderCards() {
    el.colACards.innerHTML = '';
    el.colBCards.innerHTML = '';

    if (state.blindMode) {
        // Blind mode rendering - align equivalent items on same rows
        const colAItems = getColAItems();
        const colBItems = getColBItems();
        
        // Find matches and align them
        const { alignedA, alignedB } = alignEquivalentItems(colAItems, colBItems);
        
        // Render column A
        alignedA.forEach(i => {
            if (i === null) {
                // Empty placeholder to align with Column B
                const placeholder = document.createElement('div');
                placeholder.className = 'ingredient-card-placeholder';
                el.colACards.appendChild(placeholder);
            } else {
                el.colACards.appendChild(createBlindCard(i, 'a'));
            }
        });
        
        // Render column B
        alignedB.forEach(i => {
            if (i === null) {
                // Empty placeholder to align with Column A
                const placeholder = document.createElement('div');
                placeholder.className = 'ingredient-card-placeholder';
                el.colBCards.appendChild(placeholder);
            } else {
                el.colBCards.appendChild(createBlindCard(i, 'b'));
            }
        });
        
        // Render user-added ingredients
        renderNewIngredients();
    } else {
        // Normal mode rendering
        const gtList = state.currentDish?.gt_ingredients || [];
        gtList.filter(i => state.gtVerified[i] !== 'rejected').forEach(i => el.colACards.appendChild(createGtCard(i)));
        gtList.filter(i => state.gtVerified[i] === 'rejected').forEach(i => el.colACards.appendChild(createGtCard(i)));
        state.userAdded.forEach(i => el.colACards.appendChild(createGtCard(i, true)));

        (state.currentDish?.ai_ingredients || []).forEach(i => el.colBCards.appendChild(createAiCard(i)));
    }

    updateConnectionLines();
    updateSubmitButton();
}

function renderNewIngredients() {
    if (!el.newIngredientsList) return;
    el.newIngredientsList.innerHTML = '';
    state.userAdded.forEach(ing => {
        const tag = document.createElement('div');
        tag.className = 'new-ingredient-tag';
        tag.innerHTML = `<span>${ing}</span><button onclick="removeNewIngredient('${ing.replace(/'/g, "\\'")}')">✕</button>`;
        el.newIngredientsList.appendChild(tag);
    });
}

function removeNewIngredient(ingredient) {
    state.userAdded = state.userAdded.filter(i => i !== ingredient);
    renderNewIngredients();
    updateLocalJaccard();
}

function updateConnectionLines() {
    el.connectionLines.innerHTML = '';
    const panelRect = el.annotationPanel.getBoundingClientRect();

    if (state.blindMode) {
        // Blind mode: draw links between columns
        // Lines go from external handle on Column A (right side) to external handle on Column B (left side)
        const drawBlindLine = (aIng, bIng, isAuto) => {
            const aCard = el.colACards.querySelector(`[data-ingredient="${CSS.escape(aIng)}"]`);
            const bCard = el.colBCards.querySelector(`[data-ingredient="${CSS.escape(bIng)}"]`);
            if (!aCard || !bCard) return;

            const aHandle = aCard.querySelector('.drag-handle');
            const bHandle = bCard.querySelector('.drag-handle');
            if (!aHandle || !bHandle) return;
            
            const aRect = aHandle.getBoundingClientRect();
            const bRect = bHandle.getBoundingClientRect();

            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            // From Column A's external handle (right side of card)
            line.setAttribute('x1', aRect.left + aRect.width/2 - panelRect.left);
            line.setAttribute('y1', aRect.top + aRect.height/2 - panelRect.top);
            // To Column B's external handle (left side of card)
            line.setAttribute('x2', bRect.left + bRect.width/2 - panelRect.left);
            line.setAttribute('y2', bRect.top + bRect.height/2 - panelRect.top);
            if (isAuto) line.classList.add('auto-match');
            el.connectionLines.appendChild(line);
        };

        // Draw auto-matches (fuzzy matches)
        const colAItems = getColAItems().filter(i => !state.colARejected.includes(i));
        const colBItems = getColBItems().filter(i => !state.colBRejected.includes(i));
        const drawnPairs = new Set();
        
        for (const a of colAItems) {
            for (const b of colBItems) {
                if (fuzzyMatch(a, b)) {
                    const key = `${a}|${b}`;
                    if (!drawnPairs.has(key) && !state.links.some(l => (l.a === a && l.b === b) || (l.a === b && l.b === a))) {
                        drawBlindLine(a, b, true);
                        drawnPairs.add(key);
                    }
                }
            }
        }

        // Draw manual links
        for (const link of state.links) {
            if (state.colARejected.includes(link.a) || state.colBRejected.includes(link.b)) continue;
            drawBlindLine(link.a, link.b, false);
        }
    } else {
        // Normal mode: draw AI -> GT lines
        const drawLine = (aiIng, gtIng, isAuto) => {
            const aiCard = el.colBCards.querySelector(`[data-ingredient="${CSS.escape(aiIng)}"]`);
            const gtCard = el.colACards.querySelector(`[data-ingredient="${CSS.escape(gtIng)}"]`);
            if (!aiCard || !gtCard) return;

            const handle = aiCard.querySelector('.drag-handle');
            const hRect = handle.getBoundingClientRect();
            const gRect = gtCard.getBoundingClientRect();

            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', hRect.left + hRect.width/2 - panelRect.left);
            line.setAttribute('y1', hRect.top + hRect.height/2 - panelRect.top);
            line.setAttribute('x2', gRect.right - panelRect.left);
            line.setAttribute('y2', gRect.top + gRect.height/2 - panelRect.top);
            if (isAuto) line.classList.add('auto-match');
            el.connectionLines.appendChild(line);
        };

        for (const [ai, gt] of Object.entries(state.autoMatches)) {
            if (state.aiVerified[ai] === 'rejected' || state.gtVerified[gt] === 'rejected') continue;
            if (state.equivalences[ai]?.length > 0) continue;
            drawLine(ai, gt, true);
        }

        for (const [ai, gts] of Object.entries(state.equivalences)) {
            if (state.aiVerified[ai] === 'rejected') continue;
            for (const gt of gts) {
                if (state.gtVerified[gt] === 'rejected') continue;
                drawLine(ai, gt, false);
            }
        }
    }
}

function startDrag(e, ingredient, column) {
    e.preventDefault();
    
    // Check if this item already has a MANUAL link (not auto-match)
    // If so, clicking the handle removes the link instead of starting a drag
    if (state.blindMode) {
        const existingLinkIdx = state.links.findIndex(l => l.a === ingredient || l.b === ingredient);
        if (existingLinkIdx !== -1) {
            state.links.splice(existingLinkIdx, 1);
            renderCards();
            updateLocalJaccard();
            return;
        }
    } else {
        // Normal mode: check if AI has a manual equivalence
        if (column === 'ai' && state.equivalences[ingredient]?.length > 0) {
            delete state.equivalences[ingredient];
            renderCards();
            updateLocalJaccard();
            return;
        }
    }
    
    isDragging = true;
    dragStartIngredient = ingredient;
    dragStartColumn = column;

    const rect = e.target.getBoundingClientRect();
    el.dragLine.setAttribute('x1', rect.left + rect.width/2);
    el.dragLine.setAttribute('y1', rect.top + rect.height/2);
    el.dragLine.setAttribute('x2', rect.left + rect.width/2);
    el.dragLine.setAttribute('y2', rect.top + rect.height/2);
    el.dragLineSvg.style.display = 'block';

    if (state.blindMode) {
        // Blind mode: can drop on either column (opposite to start)
        const targetSelector = column === 'a' ? '#col-b-cards' : '#col-a-cards';
        document.querySelectorAll(`${targetSelector} .ingredient-card:not(.rejected)`).forEach(c => c.classList.add('drop-target'));
    } else {
        // Normal mode: AI cards drop on GT cards
        document.querySelectorAll('#col-a-cards .ingredient-card:not(.rejected)').forEach(c => c.classList.add('drop-target'));
    }
    document.addEventListener('mousemove', onDrag);
    document.addEventListener('mouseup', endDrag);
}

function onDrag(e) {
    if (!isDragging) return;
    el.dragLine.setAttribute('x2', e.clientX);
    el.dragLine.setAttribute('y2', e.clientY);
}

function endDrag(e) {
    if (!isDragging) return;
    isDragging = false;
    el.dragLineSvg.style.display = 'none';
    document.querySelectorAll('.drop-target').forEach(c => c.classList.remove('drop-target'));
    document.removeEventListener('mousemove', onDrag);
    document.removeEventListener('mouseup', endDrag);

    const target = document.elementFromPoint(e.clientX, e.clientY);
    
    if (state.blindMode) {
        // Blind mode: bidirectional linking
        const targetSelector = dragStartColumn === 'a' ? '#col-b-cards' : '#col-a-cards';
        const targetCard = target?.closest(`${targetSelector} .ingredient-card:not(.rejected)`);
        if (targetCard) {
            const targetIng = targetCard.dataset.ingredient;
            const aIng = dragStartColumn === 'a' ? dragStartIngredient : targetIng;
            const bIng = dragStartColumn === 'b' ? dragStartIngredient : targetIng;
            // Check if link already exists
            const exists = state.links.some(l => l.a === aIng && l.b === bIng);
            if (!exists) {
                state.links.push({ a: aIng, b: bIng });
            }
            renderCards();
            updateLocalJaccard();
        }
    } else {
        // Normal mode: AI -> GT
        const gtCard = target?.closest('#col-a-cards .ingredient-card:not(.rejected)');
        if (gtCard) {
            const gtIng = gtCard.dataset.ingredient;
            if (!state.equivalences[dragStartIngredient]) state.equivalences[dragStartIngredient] = [];
            if (!state.equivalences[dragStartIngredient].includes(gtIng)) {
                state.equivalences[dragStartIngredient].push(gtIng);
            }
            renderCards();
            updateLocalJaccard();
        }
    }
    dragStartIngredient = null;
    dragStartColumn = null;
}

function calculateLocalJaccard() {
    // Start with original GT + user-added (minus rejected)
    let effectiveGt = new Set(
        [...(state.currentDish?.gt_ingredients || []), ...state.userAdded]
            .filter(i => state.gtVerified[i] !== 'rejected')
    );
    
    // ALL AI items stay in the calculation (even rejected ones)
    // Rejecting just confirms the model was wrong - it's still a false positive
    const aiList = state.currentDish?.ai_ingredients || [];
    
    // If AI ingredient is VERIFIED and doesn't match any GT,
    // add it to effective GT (the AI was right, GT was incomplete)
    for (const ai of aiList) {
        if (state.aiVerified[ai] === 'verified') {
            // Check if it matches any existing GT via fuzzy matching
            const matchesGt = [...effectiveGt].some(gt => fuzzyMatch(ai, gt));
            // Check if it has an auto-match or equivalence
            const hasAutoMatch = state.autoMatches[ai] && effectiveGt.has(state.autoMatches[ai]);
            const hasEquiv = (state.equivalences[ai] || []).some(eq => effectiveGt.has(eq));
            
            if (!matchesGt && !hasAutoMatch && !hasEquiv) {
                effectiveGt.add(ai);
            }
        }
    }

    const gtList = [...effectiveGt];
    
    if (!gtList.length && !aiList.length) return 1;
    if (!gtList.length || !aiList.length) return 0;

    let matched = 0;
    const matchedGt = new Set();

    for (const ai of aiList) {
        // Check auto-match
        const autoGt = state.autoMatches[ai];
        if (autoGt && gtList.includes(autoGt) && !matchedGt.has(autoGt)) {
            matched++;
            matchedGt.add(autoGt);
            continue;
        }
        // Check equivalence
        for (const eqGt of (state.equivalences[ai] || [])) {
            if (gtList.includes(eqGt) && !matchedGt.has(eqGt)) {
                matched++;
                matchedGt.add(eqGt);
                break;
            }
        }
        // Check fuzzy match (for verified AI that was added to GT)
        if (!matchedGt.has(ai) && gtList.includes(ai)) {
            matched++;
            matchedGt.add(ai);
        }
    }

    return matched / (gtList.length + aiList.length - matched);
}

function updateLocalJaccard() {
    el.imageUpdatedJaccard.textContent = calculateLocalJaccard().toFixed(3);
}

function hasUnresolved() {
    if (state.blindMode) {
        // Blind mode: all items must be verified/linked OR rejected
        const colAItems = getColAItems();
        const colBItems = getColBItems();
        
        // Check column A - each item must be rejected, verified, or linked
        const hasUnresolvedA = colAItems.some(i => {
            const status = getCardStatus(i, 'a');
            return status === 'unlinked'; // Not rejected, not verified, not linked
        });
        
        // Check column B - each item must be rejected, verified, or linked
        const hasUnresolvedB = colBItems.some(i => {
            const status = getCardStatus(i, 'b');
            return status === 'unlinked';
        });
        
        return hasUnresolvedA || hasUnresolvedB;
    }
    // Normal mode
    return (state.currentDish?.ai_ingredients || []).some(i => getCardStatus(i, 'ai') === 'unmatched');
}

function updateSubmitButton() {
    const unresolved = hasUnresolved();
    el.btnSubmit.disabled = unresolved;
    el.submitHint.classList.toggle('visible', unresolved);
}

function handleVerify(ingredient, type) {
    if (type === 'gt') state.gtVerified[ingredient] = 'verified';
    else state.aiVerified[ingredient] = 'verified';
    renderCards();
    updateLocalJaccard();
}

function handleReject(ingredient, type, isUserAdded = false) {
    if (type === 'gt') {
        if (isUserAdded) {
            state.userAdded = state.userAdded.filter(i => i !== ingredient);
            delete state.gtVerified[ingredient];
            for (const ai in state.equivalences) {
                state.equivalences[ai] = state.equivalences[ai].filter(g => g !== ingredient);
                if (!state.equivalences[ai].length) delete state.equivalences[ai];
            }
        } else {
            state.gtVerified[ingredient] = 'rejected';
        }
    } else {
        state.aiVerified[ingredient] = 'rejected';
        delete state.equivalences[ingredient];
    }
    renderCards();
    updateLocalJaccard();
}

function handleAddIngredient() {
    const val = el.newIngredientInput.value.trim();
    if (val && !state.userAdded.includes(val)) {
        state.userAdded.push(val);
        state.gtVerified[val] = 'verified';
        el.newIngredientInput.value = '';
        renderCards();
        updateLocalJaccard();
    }
}

function handleReset() {
    // Reset all user changes back to initial state for current dish
    if (!state.currentDish) return;
    
    // Clear all verification states
    state.gtVerified = {};
    state.aiVerified = {};
    state.equivalences = {};
    state.userAdded = [];
    state.challengeFlag = false;
    
    // Clear blind mode state
    state.colARejected = [];
    state.colBRejected = [];
    state.links = [];
    
    // Recalculate auto-matches from original data
    state.autoMatches = state.currentDish.auto_matches || {};
    
    // Re-render cards
    renderCards();
    if (state.blindMode) renderNewIngredients();
    updateLocalJaccard();
}

async function handleSubmit() {
    if (!state.currentDish) return;
    
    // Single-blind mode handling
    if (state.singleBlindMode) {
        await handleSingleBlindSubmit();
        return;
    }
    
    if (hasUnresolved()) return;
    showLoading(true);

    try {
        const payload = {
            dataset: state.currentDataset,
            image_id: state.currentDish.image_id,
            annotator: state.currentPlayer || '',
            gt_verified: state.gtVerified,
            ai_verified: state.aiVerified,
            equivalences: state.equivalences,
            user_added: state.userAdded,
            challenge_flag: state.challengeFlag,
            // Blind mode fields
            column_a_is_gt: state.columnAIsGt,
            col_a_rejected: state.colARejected,
            col_b_rejected: state.colBRejected,
            col_a_verified: state.colAVerified,
            col_b_verified: state.colBVerified,
            links: state.links
        };
        
        // In blind mode, convert links to equivalences format for backend
        if (state.blindMode) {
            // Decode which column was GT and convert accordingly
            const newEquivalences = {};
            for (const link of state.links) {
                const aiIng = state.columnAIsGt ? link.b : link.a;
                const gtIng = state.columnAIsGt ? link.a : link.b;
                if (!newEquivalences[aiIng]) newEquivalences[aiIng] = [];
                if (!newEquivalences[aiIng].includes(gtIng)) {
                    newEquivalences[aiIng].push(gtIng);
                }
            }
            payload.equivalences = newEquivalences;
            
            // Convert column rejections to gt/ai verified format
            payload.gt_verified = {};
            payload.ai_verified = {};
            const gtRejected = state.columnAIsGt ? state.colARejected : state.colBRejected;
            const aiRejected = state.columnAIsGt ? state.colBRejected : state.colARejected;
            gtRejected.forEach(i => payload.gt_verified[i] = 'rejected');
            aiRejected.forEach(i => payload.ai_verified[i] = 'rejected');
        }
        
        await submitAnnotation(payload);

        const stats = await fetchStats(state.currentDataset);
        updateStats(stats);
        await loadNextImage();
    } catch (err) {
        console.error(err);
        showLoading(false);
    }
}

async function handleSingleBlindSubmit() {
    showLoading(true);
    
    try {
        // Categorize by source (gt, ai, both, user)
        const gt_approved = [], gt_rejected = [], gt_unsure = [];
        const ai_approved = [], ai_rejected = [], ai_unsure = [];
        
        for (const item of state.mergedIngredients) {
            const name = item.name;
            const source = item.source;  // 'gt', 'ai', 'both', or 'user'
            
            // Skip user-added (handled separately)
            if (source === 'user') continue;
            
            const isApproved = state.approvedIngredients.includes(name);
            const isRejected = state.rejectedIngredients.includes(name);
            const isUnsure = state.unsureIngredients.includes(name);
            
            // Items with source 'both' go to both GT and AI lists
            if (source === 'gt' || source === 'both') {
                if (isApproved) gt_approved.push(item.gt_name || name);
                else if (isRejected) gt_rejected.push(item.gt_name || name);
                else if (isUnsure) gt_unsure.push(item.gt_name || name);
            }
            if (source === 'ai' || source === 'both') {
                if (isApproved) ai_approved.push(item.ai_name || name);
                else if (isRejected) ai_rejected.push(item.ai_name || name);
                else if (isUnsure) ai_unsure.push(item.ai_name || name);
            }
        }
        
        const payload = {
            dataset: state.currentDataset,
            image_id: state.currentDish.image_id,
            annotator: state.currentPlayer || '',
            phase: 'phase1',
            gt_approved, gt_rejected, gt_unsure,
            ai_approved, ai_rejected, ai_unsure,
            user_added: state.userAdded
        };
        
        await submitAnnotation(payload);
        
        const stats = await fetchStats(state.currentDataset);
        updateStats(stats);
        await loadNextImage();
    } catch (err) {
        console.error(err);
        showLoading(false);
    }
}

async function loadNextImage() {
    showLoading(true);
    resetState();

    try {
        const data = await fetchNext(state.currentDataset, state.currentSet);
        if (data.done || data.error) {
            // If main set is done, check if we should prompt for challenge set
            if (state.currentSet === 'main' && state.stats.challenge_count > 0) {
                showDoneWithChallenge();
            } else {
                showDone(state.stats.updated_jaccard_mean);
            }
            return;
        }

        state.currentDish = data;
        state.autoMatches = data.auto_matches || {};
        state.columnAIsGt = data.column_a_is_gt !== false; // Default true if not specified

        el.currentImage.textContent = data.image_id;
        el.foodImage.src = data.image_url;
        el.aiDescriptionText.textContent = data.ai_description || 'No description';
        el.imageOriginalJaccard.textContent = (data.original_jaccard || 0).toFixed(3);
        el.imageUpdatedJaccard.textContent = (data.original_jaccard || 0).toFixed(3);

        // Challenge status is now determined by unsure ingredients, not a manual flag

        if (data.existing_annotation) {
            const ann = data.existing_annotation;
            state.gtVerified = ann.gt_verified || {};
            state.aiVerified = ann.ai_verified || {};
            state.equivalences = ann.equivalences || {};
            state.userAdded = ann.user_added || [];
            state.challengeFlag = !!ann.challenge_flag;
            
            // Restore blind mode state if applicable
            if (ann.column_a_is_gt !== undefined) {
                state.columnAIsGt = ann.column_a_is_gt;
            }
            
            // Restore single-blind mode state if applicable (best-effort; normally completed images are skipped)
            state.approvedIngredients = ann.approved_ingredients || [];
            state.rejectedIngredients = ann.rejected_ingredients || [];
            state.unsureIngredients = ann.unsure_ingredients || [];
        }

        // Single-blind mode: load merged ingredients
        if (state.singleBlindMode && data.merged_ingredients) {
            state.mergedIngredients = data.merged_ingredients;
            // Add user-added ingredients to merged list
            state.userAdded.forEach(ing => {
                if (!state.mergedIngredients.some(m => m.name === ing)) {
                    state.mergedIngredients.push({
                        name: ing,
                        gt_name: null,
                        ai_name: null,
                        source: 'user',
                        fuzzy_match: false
                    });
                }
            });
            
            // Ensure Phase 1 panel is visible
            if (el.singleBlindPhase1) el.singleBlindPhase1.style.display = 'flex';
            if (el.globalPhase2) el.globalPhase2.style.display = 'none';
            const submitTextEl = el.btnSubmit?.querySelector('span:last-child');
            if (submitTextEl) submitTextEl.textContent = 'Submit & Next';
            
            renderSingleBlindPhase1();
            updateSingleBlindSubmitButton();
        } else {
            renderCards();
            updateLocalJaccard();
        }
    } catch (err) {
        console.error(err);
    } finally {
        showLoading(false);
    }
}

async function changeDataset() {
    const ds = el.datasetSelect.value;
    if (!ds) return;
    state.currentDataset = ds;

    // Only update badge in normal mode
    if (!state.blindMode) {
        el.colBBadge.textContent = ds === 'foodseg103' ? 'Gemini' : 'Gemini 2.0 Flash';
    }

    showLoading(true);
    try {
        const stats = await fetchStats(ds);
        updateStats(stats);
        await loadNextImage();
    } catch (err) {
        console.error(err);
        showLoading(false);
    }
}

function setupBlindMode() {
    document.body.classList.add('blind-mode');
    el.colATitle.textContent = 'Column A';
    el.colBTitle.textContent = 'Column B';
    el.colABadge.textContent = 'Source 1';
    el.colBBadge.textContent = 'Source 2';
    el.colABadge.className = 'column-badge';
    el.colBBadge.className = 'column-badge';
    
    // Hide AI description in blind mode (would give away which is AI)
    if (el.aiDescriptionSection) el.aiDescriptionSection.style.display = 'none';
    
    // Hide Jaccard displays in blind mode
    document.querySelectorAll('.jaccard-compare').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.stat-label').forEach(el => {
        if (el.textContent.includes('Jaccard') || el.textContent.includes('This Image')) {
            el.parentElement.style.display = 'none';
        }
    });
    
    // Show new ingredients section, hide old add ingredient
    if (el.newIngredientsSection) el.newIngredientsSection.style.display = 'block';
    if (el.addIngredientSection) el.addIngredientSection.style.display = 'none';
}

function setupNormalMode() {
    document.body.classList.remove('blind-mode');
    el.colATitle.textContent = 'Ground Truth';
    el.colBTitle.textContent = 'AI Predictions';
    el.colABadge.textContent = 'Original Labels';
    el.colABadge.className = 'column-badge gt-badge';
    
    // Hide new ingredients section, show add ingredient
    if (el.newIngredientsSection) el.newIngredientsSection.style.display = 'none';
    if (el.addIngredientSection) el.addIngredientSection.style.display = 'flex';
}

function handleBlindAddIngredient() {
    const input = el.blindNewIngredientInput;
    const val = input.value.trim();
    if (val && !state.userAdded.includes(val)) {
        state.userAdded.push(val);
        input.value = '';
        renderNewIngredients();
        updateLocalJaccard();
    }
}

// ============================================
// Dashboard Functions
// ============================================

let precisionChartInstance = null;
let breakdownChartInstance = null;

async function loadAnnotators() {
    if (!state.currentDataset || !el.annotatorFilterSelect) return;
    
    try {
        const annotators = await fetch(`api/annotators/${state.currentDataset}`).then(r => r.json());
        
        // Save current selection
        const currentValue = el.annotatorFilterSelect.value;
        
        // Clear and rebuild options
        el.annotatorFilterSelect.innerHTML = '<option value="">All</option>';
        annotators.forEach(name => {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            el.annotatorFilterSelect.appendChild(option);
        });
        
        // Restore selection if still valid
        if (annotators.includes(currentValue)) {
            el.annotatorFilterSelect.value = currentValue;
        }
    } catch (err) {
        console.error('Failed to load annotators:', err);
    }
}

async function showDashboard() {
    if (!state.currentDataset) return;
    
    // Show dashboard, hide main content
    el.dashboardPanel.style.display = 'flex';
    el.mainContent.style.display = 'none';
    document.body.classList.add('dashboard-active');
    
    // Load annotators for filter dropdown
    await loadAnnotators();
    
    // Fetch and display stats
    await refreshDashboardStats();
}

async function refreshDashboardStats() {
    if (!state.currentDataset) return;
    
    // Default to excluding perfect if toggle not found (or use toggle state)
    const excludePerfect = el.excludePerfectToggle ? (el.excludePerfectToggle.checked ? '1' : '0') : '1';
    const annotator = el.annotatorFilterSelect ? el.annotatorFilterSelect.value : '';
    
    try {
        let url = `api/dashboard/${state.currentDataset}?exclude_perfect=${excludePerfect}`;
        if (annotator) {
            url += `&annotator=${encodeURIComponent(annotator)}`;
        }
        const stats = await fetch(url).then(r => r.json());
        renderDashboardStats(stats, el.excludePerfectToggle?.checked);
        renderDashboardCharts(stats);
    } catch (err) {
        console.error('Failed to load dashboard stats:', err);
    }
}

function hideDashboard() {
    el.dashboardPanel.style.display = 'none';
    el.mainContent.style.display = '';
    document.body.classList.remove('dashboard-active');
}

function renderDashboardStats(stats, excludePerfect = false) {
    const { counts, gt, ai, cutoff_date } = stats;
    
    // Calculate images that need manual annotation (exclude auto-perfect)
    const needsAnnotation = counts.total_images_in_dataset - counts.auto_perfect;
    
    // Completed photos - apply toggle
    const displayedCount = excludePerfect ? counts.manual : counts.total;
    el.dashCompletedCount.textContent = displayedCount;
    el.dashCompletedDetail.textContent = excludePerfect 
        ? `Manual only (${counts.auto_perfect} auto-perfect excluded)`
        : `Manual: ${counts.manual} | Auto-Perfect: ${counts.auto_perfect}`;
    
    // Progress - apply toggle (denominator excludes auto-perfect when toggle is on)
    const progressNumerator = excludePerfect ? counts.manual : counts.total;
    const progressDenominator = excludePerfect ? needsAnnotation : counts.total_images_in_dataset;
    const progressPct = progressDenominator > 0 
        ? Math.round((progressNumerator / progressDenominator) * 100)
        : 0;
    el.dashProgressPct.textContent = `${progressPct}%`;
    el.dashProgressDetail.textContent = excludePerfect
        ? `${counts.manual} / ${needsAnnotation} images needing annotation`
        : `${counts.total} / ${counts.total_images_in_dataset} images`;
    
    // GT Precision
    const gtTotal = gt.approved + gt.rejected + gt.unsure;
    el.dashGtPrecisionStrict.textContent = `${(gt.precision_strict * 100).toFixed(1)}%`;
    el.dashGtPrecisionUncertain.textContent = `${(gt.precision_with_uncertain * 100).toFixed(1)}%`;
    el.dashGtApproved.textContent = gtTotal > 0 ? `${gt.approved}/${gtTotal} (${(gt.approved/gtTotal*100).toFixed(1)}%)` : '--';
    el.dashGtRejected.textContent = gtTotal > 0 ? `${gt.rejected}/${gtTotal} (${(gt.rejected/gtTotal*100).toFixed(1)}%)` : '--';
    el.dashGtUnsure.textContent = gtTotal > 0 ? `${gt.unsure}/${gtTotal} (${(gt.unsure/gtTotal*100).toFixed(1)}%)` : '--';
    
    // AI Precision
    const aiTotal = ai.approved + ai.rejected + ai.unsure;
    el.dashAiPrecisionStrict.textContent = `${(ai.precision_strict * 100).toFixed(1)}%`;
    el.dashAiPrecisionUncertain.textContent = `${(ai.precision_with_uncertain * 100).toFixed(1)}%`;
    el.dashAiApproved.textContent = aiTotal > 0 ? `${ai.approved}/${aiTotal} (${(ai.approved/aiTotal*100).toFixed(1)}%)` : '--';
    el.dashAiRejected.textContent = aiTotal > 0 ? `${ai.rejected}/${aiTotal} (${(ai.rejected/aiTotal*100).toFixed(1)}%)` : '--';
    el.dashAiUnsure.textContent = aiTotal > 0 ? `${ai.unsure}/${aiTotal} (${(ai.unsure/aiTotal*100).toFixed(1)}%)` : '--';
    
    // Cutoff date
    el.dashCutoffDate.textContent = cutoff_date;
}

function renderDashboardCharts(stats) {
    const { gt, ai } = stats;
    
    // Chart.js color scheme matching our CSS variables
    const colors = {
        green: '#6bba6b',
        red: '#e85a5a',
        yellow: '#e8c85a',
        orange: '#e8945a',
        blue: '#5a9ae8',
        text: '#a89a8a',
        bg: '#2d2620'
    };
    
    // Destroy existing charts if they exist
    if (precisionChartInstance) {
        precisionChartInstance.destroy();
    }
    if (breakdownChartInstance) {
        breakdownChartInstance.destroy();
    }
    
    // Precision Comparison Bar Chart
    const precisionCtx = el.precisionChart.getContext('2d');
    precisionChartInstance = new Chart(precisionCtx, {
        type: 'bar',
        data: {
            labels: ['N5K Ground Truth', 'AI (Gemini 2.0)'],
            datasets: [
                {
                    label: 'Strict Precision',
                    data: [gt.precision_strict * 100, ai.precision_strict * 100],
                    backgroundColor: colors.green,
                    borderRadius: 4
                },
                {
                    label: 'With Uncertain',
                    data: [gt.precision_with_uncertain * 100, ai.precision_with_uncertain * 100],
                    backgroundColor: colors.yellow,
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: { color: colors.text }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { 
                        color: colors.text,
                        callback: value => value + '%'
                    },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                x: {
                    ticks: { color: colors.text },
                    grid: { display: false }
                }
            }
        }
    });
    
    // Ingredient Breakdown Stacked Bar Chart
    const breakdownCtx = el.breakdownChart.getContext('2d');
    breakdownChartInstance = new Chart(breakdownCtx, {
        type: 'bar',
        data: {
            labels: ['N5K Ground Truth', 'AI (Gemini 2.0)'],
            datasets: [
                {
                    label: 'Approved',
                    data: [gt.approved, ai.approved],
                    backgroundColor: colors.green,
                    borderRadius: 4
                },
                {
                    label: 'Rejected',
                    data: [gt.rejected, ai.rejected],
                    backgroundColor: colors.red,
                    borderRadius: 4
                },
                {
                    label: 'Unsure',
                    data: [gt.unsure, ai.unsure],
                    backgroundColor: colors.yellow,
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: { color: colors.text }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    stacked: true,
                    ticks: { color: colors.text },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                x: {
                    stacked: true,
                    ticks: { color: colors.text },
                    grid: { display: false }
                }
            }
        }
    });
}

async function init() {
    el.btnReset.onclick = handleReset;
    el.btnSkip.onclick = () => loadNextImage();
    el.btnSubmit.onclick = handleSubmit;
    el.btnAddIngredient.onclick = handleAddIngredient;
    el.newIngredientInput.onkeypress = (e) => { if (e.key === 'Enter') handleAddIngredient(); };
    el.datasetSelect.onchange = changeDataset;
    
    // Global Phase 2 handlers (single-blind mode)
    if (el.btnProceedPhase2) {
        el.btnProceedPhase2.onclick = () => loadGlobalPhase2();
    }
    if (el.btnPhase2Finish) {
        el.btnPhase2Finish.onclick = () => handlePhase2Finish();
    }
    
    // Set toggle handlers (main vs challenge set)
    if (el.btnMainSet) {
        el.btnMainSet.onclick = () => switchToSet('main');
    }
    if (el.btnChallengeSet) {
        el.btnChallengeSet.onclick = () => switchToSet('challenge');
    }
    
    // Dashboard button handler
    if (el.btnDashboard) {
        el.btnDashboard.onclick = () => showDashboard();
    }
    
    // Dashboard back button handler
    if (el.btnDashboardBack) {
        el.btnDashboardBack.onclick = () => hideDashboard();
    }
    
    // Dashboard exclude perfect toggle handler
    if (el.excludePerfectToggle) {
        el.excludePerfectToggle.onchange = () => refreshDashboardStats();
    }
    
    // Dashboard annotator filter handler
    if (el.annotatorFilterSelect) {
        el.annotatorFilterSelect.onchange = () => refreshDashboardStats();
    }
    
    // Export button handler (inside dashboard)
    if (el.btnExport) {
        el.btnExport.onclick = () => {
            if (state.currentDataset) {
                window.location.href = `api/export/${state.currentDataset}?include_perfect=1`;
            }
        };
    }
    
    // Blind mode add ingredient handlers
    if (el.btnBlindAddIngredient) {
        el.btnBlindAddIngredient.onclick = handleBlindAddIngredient;
    }
    if (el.blindNewIngredientInput) {
        el.blindNewIngredientInput.onkeypress = (e) => { if (e.key === 'Enter') handleBlindAddIngredient(); };
    }

    window.addEventListener('resize', updateConnectionLines);
    el.colACards.addEventListener('scroll', updateConnectionLines);
    el.colBCards.addEventListener('scroll', updateConnectionLines);

    // Setup player selection
    setupPlayerSelection();
    
    // Setup magnifier for image zoom
    setupMagnifier();

    showLoading(true);
    try {
        // Fetch config to check blind mode
        const config = await fetchConfig();
        state.blindMode = config.blind_mode || false;
        state.singleBlindMode = config.single_blind_mode || false;
        console.log('Blind mode:', state.blindMode);
        console.log('Single-blind mode:', state.singleBlindMode);
        
        if (state.singleBlindMode) {
            setupSingleBlindMode();
        } else if (state.blindMode) {
            setupBlindMode();
        } else {
            setupNormalMode();
        }
        
        const datasets = await fetchDatasets();
        console.log('Available datasets:', datasets);
        
        // Build dropdown, showing all datasets but disabling those with no images
        el.datasetSelect.innerHTML = datasets.map(d => {
            const hasImages = d.total_images > 0;
            const label = hasImages 
                ? `${d.name} (${d.total_images} images)`
                : `${d.name} (no images found)`;
            return `<option value="${d.key}" ${!hasImages ? 'disabled' : ''}>${label}</option>`;
        }).join('');
        
        // Select first dataset with images
        const firstWithImages = datasets.find(d => d.total_images > 0);
        if (firstWithImages) {
            el.datasetSelect.value = firstWithImages.key;
            state.currentDataset = firstWithImages.key;
            const stats = await fetchStats(state.currentDataset);
            updateStats(stats);
            await loadNextImage();
            
            // Now that dataset is loaded, check for saved player
            if (!loadSavedPlayer()) {
                showPlayerModal();
            }
        } else {
            el.datasetSelect.innerHTML = '<option disabled>No datasets available</option>';
            showLoading(false);
        }
    } catch (err) {
        console.error(err);
    }
}

// Make removeNewIngredient available globally for onclick handler
window.removeNewIngredient = removeNewIngredient;

document.addEventListener('DOMContentLoaded', init);
