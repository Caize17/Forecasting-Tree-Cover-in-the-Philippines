/* =========================================================
   CONFIG
========================================================= */
const API_BASE_URL = 'https://forecasting-tree-cover-in-the-philippines.onrender.com/api';

/* =========================================================
   MOBILE MENU
========================================================= */
const menuToggle = document.querySelector('.menu-toggle');
const mobileMenu = document.querySelector('#mobile-menu');

menuToggle.addEventListener('click', () => {
    const isExpanded = menuToggle.getAttribute('aria-expanded') === 'true';
    menuToggle.setAttribute('aria-expanded', (!isExpanded).toString());
    mobileMenu.hidden = isExpanded;
    menuToggle.classList.toggle('open');
});

document.querySelectorAll('.mobile-link').forEach(link => {
    link.addEventListener('click', () => {
        mobileMenu.hidden = true;
        menuToggle.classList.remove('open');
        menuToggle.setAttribute('aria-expanded', 'false');
    });
});

/* =========================================================
   NAVIGATION ACTIVE LINK
========================================================= */
const sections = document.querySelectorAll('.section');
const navLinks = document.querySelectorAll('.nav-link');

function updateActiveLink() {
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        if (window.scrollY >= sectionTop - 200) {
            current = section.getAttribute('id');
        }
    });
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
}

window.addEventListener('scroll', updateActiveLink);
window.addEventListener('load', updateActiveLink);

/* =========================================================
   SMOOTH SCROLL
========================================================= */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            const headerOffset = 80;
            const elementPosition = target.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

            window.scrollTo({ top: offsetPosition, behavior: 'smooth' });
        }
    });
});

/* =========================================================
   API WRAPPER
========================================================= */
async function apiGet(endpoint) {
    try {
        const res = await fetch(`${API_BASE_URL}${endpoint}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error(`Error fetching ${endpoint}:`, err);
        return null;
    }
}

/* =========================================================
   GLOBAL DATA
========================================================= */
let globalData = {
    forecasts: [],
    comparison: [],
    drivers: [],
    regions: []
};

/* =========================================================
   INITIALIZER
========================================================= */
async function initializeData() {
    try {
        globalData.drivers = await apiGet('/drivers');
        globalData.forecasts = await apiGet('/forecasts');
        globalData.comparison = await apiGet('/comparison');

        globalData.regions = [...new Set(globalData.forecasts.map(f => f.Region))].sort();

        populateRegionSelects();
        loadAccuracyTable();

        loadDriverCoefficients();
        loadDriverHistory();

        console.log('Data initialized successfully:', globalData);

    } catch (err) {
        showError('Failed to load data. Make sure backend is running.');
    }
}

/* =========================================================
   DROPDOWN POPULATOR
========================================================= */
function populateRegionSelects() {
    const regionSelect = document.getElementById('region-select');
    const driverRegionSelect = document.getElementById('driver-region-select');
    const accuracyRegionSelect = document.getElementById('accuracy-region-select');

    const opts = globalData.regions.map(r => `<option value="${r}">${r}</option>`).join('');

    if (regionSelect) regionSelect.innerHTML = `<option value="">Select a region...</option>${opts}`;
    if (driverRegionSelect) driverRegionSelect.innerHTML = `<option value="">Select a region...</option>${opts}`;
    if (accuracyRegionSelect) accuracyRegionSelect.innerHTML = `<option value="">All Regions</option>${opts}`;
}

/* =========================================================
   ERROR POPUP
========================================================= */
function showError(message) {
    const div = document.createElement('div');
    div.style.cssText = 'position:fixed;top:100px;left:50%;transform:translateX(-50%);background:#f44336;color:white;padding:20px;border-radius:8px;z-index:9999;';
    div.textContent = message;
    document.body.appendChild(div);
    setTimeout(() => div.remove(), 4000);
}

/* =========================================================
   FORECAST & COMPARISON CHARTS
========================================================= */

let chartInstance = null;

function renderForecastChart(region) {
    const data = globalData.forecasts.filter(f => f.Region === region);
    if (!data.length) return showError('No forecast data available.');

    renderChart(
        data.map(f => f.Year),
        [{
            label: 'SARIMAX Forecast',
            data: data.map(f => f.SARIMAX_Forecast),
            borderColor: '#59855C',
            backgroundColor: 'rgba(89,133,92,0.1)',
            borderWidth: 3,
            tension: 0.4,
            fill: true
        }]
    );
}

function renderComparisonChart(region) {
    const data = globalData.comparison.filter(c => c.Region === region);
    if (!data.length) return showError('No comparison data available.');

    renderChart(
        data.map(c => c.Year),
        [
            {
                label: 'Actual',
                data: data.map(c => c.Actual),
                borderColor: '#2fa44f',
                backgroundColor: 'rgba(47,164,79,0.1)',
                borderWidth: 3,
                tension: 0.4,
                fill: true
            },
            {
                label: 'Forecast',
                data: data.map(c => c.SARIMAX_Forecast),
                borderColor: '#f0a84b',
                backgroundColor: 'rgba(240,168,75,0.1)',
                borderDash: [6, 4],
                borderWidth: 3,
                tension: 0.4,
                fill: true
            }
        ]
    );
}

function renderChart(labels, datasets) {
    const container = document.getElementById('chart-container');
    container.innerHTML = `<canvas id="main-chart"></canvas>`;

    const ctx = document.getElementById('main-chart').getContext('2d');

    if (chartInstance) chartInstance.destroy();

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: {
                    ticks: { callback: v => v.toLocaleString() }
                }
            }
        }
    });
}

/* =========================================================
   DRIVER COEFFICIENTS (from drivers.js)
========================================================= */

const coefsContainer = document.getElementById('drivers-coefs');

function loadDriverCoefficients() {
    if (!globalData.drivers || !Array.isArray(globalData.drivers)) return;

    const byRegion = {};

    globalData.drivers.forEach(row => {
        if (!byRegion[row.Region]) byRegion[row.Region] = [];
        byRegion[row.Region].push(row);
    });

    for (const region in byRegion) {
        const block = document.createElement('div');
        block.className = 'region-block';
        block.innerHTML = `<div class="region-title">${region}</div>`;

        byRegion[region].forEach(r => {
            block.innerHTML += `
                <div class="coef-row">
                    <span>${r.Driver}</span>
                    <span>Coef: ${Number(r.Coefficient).toFixed(4)}</span>
                    <span>P-value: ${Number(r.PValue).toExponential(2)}</span>
                </div>`;
        });

        coefsContainer.appendChild(block);
    }
}

/* =========================================================
   DRIVER HISTORY SECTION (fallback included)
========================================================= */

const fallbackDriverData = {
    "Logging":[{year:2015,value:22},{year:2016,value:23},{year:2017,value:24},{year:2018,value:25},{year:2019,value:26},{year:2020,value:25},{year:2021,value:24},{year:2022,value:25},{year:2023,value:26},{year:2024,value:25}],
    "Agriculture":[{year:2015,value:38},{year:2016,value:39},{year:2017,value:39},{year:2018,value:40},{year:2019,value:41},{year:2020,value:40},{year:2021,value:39},{year:2022,value:40},{year:2023,value:41},{year:2024,value:40}],
    "Urbanization":[{year:2015,value:13},{year:2016,value:14},{year:2017,value:14},{year:2018,value:15},{year:2019,value:16},{year:2020,value:15},{year:2021,value:14},{year:2022,value:15},{year:2023,value:16},{year:2024,value:15}],
    "Mining":[{year:2015,value:9},{year:2016,value:9},{year:2017,value:10},{year:2018,value:10},{year:2019,value:11},{year:2020,value:10},{year:2021,value:9},{year:2022,value:10},{year:2023,value:11},{year:2024,value:10}]
};

function loadDriverHistory() {
    const container = document.getElementById('driver-data-viewer');
    const dataGrid = document.getElementById('driver-data-grid');

    const drivers = fallbackDriverData; // ← you can replace with real API later

    const driverNames = Object.keys(drivers);
    const selector = document.createElement('select');

    driverNames.forEach(d => {
        const opt = document.createElement('option');
        opt.value = d;
        opt.textContent = d;
        selector.appendChild(opt);
    });

    selector.addEventListener('change', () => {
        renderDriverHistory(drivers[selector.value]);
    });

    container.insertBefore(selector, dataGrid);

    renderDriverHistory(drivers[driverNames[0]]);
}

function renderDriverHistory(rows) {
    const grid = document.getElementById('driver-data-grid');
    grid.innerHTML = '';

    rows.forEach(r => {
        grid.innerHTML += `
            <div class="driver-year-data">
                <div class="year">${r.year}</div>
                <div class="value">${r.value}%</div>
            </div>`;
    });
}

/* =========================================================
   ACCURACY TABLE
========================================================= */
function loadAccuracyTable(filter = '') {
    const box = document.getElementById('accuracy-table-container');

    let data = globalData.comparison;
    if (filter) data = data.filter(r => r.Region === filter);

    if (!data.length) {
        box.innerHTML = `<p class="muted-text">No data available</p>`;
        return;
    }

    let html = `
        <table class="accuracy-table">
        <thead>
            <tr>
                <th>Region</th>
                <th>Year</th>
                <th>Actual</th>
                <th>Forecast</th>
                <th>MAE</th>
                <th>MAPE</th>
            </tr>
        </thead><tbody>
    `;

    data.forEach(r => {
        html += `
            <tr>
                <td>${r.Region}</td>
                <td>${r.Year}</td>
                <td>${r.Actual.toLocaleString()}</td>
                <td>${r.SARIMAX_Forecast.toLocaleString()}</td>
                <td>${r.AE.toLocaleString()}</td>
                <td>${r.APE.toFixed(2)}%</td>
            </tr>`;
    });

    html += `</tbody></table>`;
    box.innerHTML = html;
}

/* =========================================================
   FOOTER YEAR
========================================================= */
document.querySelector('#year').textContent = new Date().getFullYear();

/* =========================================================
   INIT APP
========================================================= */
document.addEventListener('DOMContentLoaded', () => {
    updateActiveLink();
    initializeData();
    console.log('ForestCastPH initialized');
});
