document.addEventListener('DOMContentLoaded', async function () {
  const labelSel = document.getElementById('label');
  const patternCountEl = document.getElementById('patternCount');

  async function updateCount() {
    const label = labelSel.value;
    if (!label) return;
    try {
      const res = await fetch(`/count/${encodeURIComponent(label)}`);
      const data = await res.json();
      patternCountEl.textContent = `Сохранено: ${data.count}`;
    } catch (err) {
      console.error('Ошибка получения счётчика', err);
    }
  }

  try {
    const res = await fetch('settings.json');
    const data = await res.json();
    if (Array.isArray(data.labels)) {
      data.labels.forEach((l) => {
        const opt = document.createElement('option');
        opt.value = l;
        opt.textContent = l;
        labelSel.appendChild(opt);
      });
      if (data.labels.length > 0) {
        labelSel.value = data.labels[0];
        updateCount();
      }
    }
  } catch (e) {
    console.error('Ошибка загрузки settings.json', e);
  }

  labelSel.addEventListener('change', updateCount);

  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const brush = document.getElementById('brush');
  const saveBtn = document.getElementById('saveBtn');
  const clearBtn = document.getElementById('clearBtn');
  const statusEl = document.getElementById('status');

  function resizeCanvas() {
    const dpr = Math.max(window.devicePixelRatio || 1, 1);
    const rect = canvas.getBoundingClientRect();
    const size = Math.min(rect.width, rect.height || rect.width);
    const target = Math.max(200, Math.floor(size));
    canvas.width = Math.floor(target * dpr);
    canvas.height = Math.floor(target * dpr);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = parseInt(brush.value, 10);
  }

  brush.addEventListener('input', () => {
    ctx.lineWidth = parseInt(brush.value, 10);
  });

  let drawing = false;
  let lastX = 0;
  let lastY = 0;

  function startDraw(x, y) {
    drawing = true;
    lastX = x;
    lastY = y;
  }

  function drawLine(x, y) {
    if (!drawing) return;
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    lastX = x;
    lastY = y;
  }

  function stopDraw() {
    drawing = false;
  }

  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    if (e.touches && e.touches.length > 0) {
      return {
        x: e.touches[0].clientX - rect.left,
        y: e.touches[0].clientY - rect.top
      };
    } else {
      return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      };
    }
  }

  function init() {
    requestAnimationFrame(() => {
      const parent = canvas.parentElement.getBoundingClientRect();
      canvas.style.width = parent.width + 'px';
      canvas.style.height = parent.width + 'px';
      resizeCanvas();

      canvas.addEventListener('mousedown', (e) => {
        const pos = getPos(e);
        startDraw(pos.x, pos.y);
      });
      canvas.addEventListener('mousemove', (e) => {
        const pos = getPos(e);
        drawLine(pos.x, pos.y);
      });
      canvas.addEventListener('mouseup', stopDraw);
      canvas.addEventListener('mouseleave', stopDraw);

      canvas.addEventListener('touchstart', (e) => {
        const pos = getPos(e);
        startDraw(pos.x, pos.y);
      });
      canvas.addEventListener('touchmove', (e) => {
        const pos = getPos(e);
        drawLine(pos.x, pos.y);
        e.preventDefault();
      }, { passive: false });
      canvas.addEventListener('touchend', stopDraw);

      clearBtn.addEventListener('click', () => {
        resizeCanvas();
      });

      saveBtn.addEventListener('click', async () => {
        const imageDataURL = canvas.toDataURL('image/png');
        const label = labelSel.value;
        try {
          const res = await fetch('/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageDataURL, label })
          });
          const result = await res.json();
          if (result.status === 'ok') {
            statusEl.textContent = `Рисунок «${label}» успешно сохранён`;
            updateCount();
            if (typeof result.count === 'number') {
              patternCountEl.textContent = `Сохранено: ${result.count}`;
            }
            resizeCanvas();
            setTimeout(() => statusEl.textContent = '', 3000);
          } else {
            statusEl.textContent = 'Ошибка сохранения';
          }
        } catch (err) {
          console.error(err);
          statusEl.textContent = 'Ошибка сети';
        }
      });
    });
  }

  init();
});
