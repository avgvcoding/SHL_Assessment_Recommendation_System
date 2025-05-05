const form = document.getElementById('recommend-form');
const input = document.getElementById('query-input');
const loading = document.getElementById('loading');
const table = document.getElementById('results-table');
const tbody = document.getElementById('results-body');

const API_BASE = 'https://shl-assessment-recommendation-system-z06m.onrender.com';

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  loading.classList.remove('hidden');
  table.classList.add('hidden');
  tbody.innerHTML = '';

  try {
    const res = await fetch(`${API_BASE}/recommend`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ query })
    });
    if (!res.ok) throw new Error('Network response was not OK');

    const data = await res.json();

    data.recommended_assessments.forEach(item => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td><a href="${item.url}" target="_blank">${item.url}</a></td>
        <td>${item.duration} min</td>
        <td>${item.remote_support}</td>
        <td>${item.adaptive_support}</td>
        <td>${item.test_type.join(', ')}</td>
      `;
      tbody.appendChild(row);
    });

    table.classList.remove('hidden');
  } catch (err) {
    alert('Error fetching recommendations: ' + err.message);
  } finally {
    loading.classList.add('hidden');
  }
});