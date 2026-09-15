document.addEventListener("DOMContentLoaded", () => {
  const data = JSON.parse(document.getElementById("spike-data").textContent);
  const { sessionId, players, spikes } = data;

  const spikesByPlayer = {};
  players.forEach((p) => { spikesByPlayer[p.id] = []; });
  spikes.forEach((s) => {
    if (!spikesByPlayer[s.player_id]) spikesByPlayer[s.player_id] = [];
    spikesByPlayer[s.player_id].push(s);
  });
  Object.values(spikesByPlayer).forEach((list) => list.sort((a, b) => a.player_spike_number - b.player_spike_number));

  function findSpike(playerId, count) {
    return (spikesByPlayer[playerId] || []).find((s) => s.player_spike_number === count);
  }

  function createPanel(side, initialCount) {
    const panelEl = document.querySelector(`.video-panel[data-side="${side}"]`);
    const navEl = document.querySelector(`.nav-buttons[data-side="${side}"]`);
    const playerSelect = panelEl.querySelector(".player-select");
    const countSelect = panelEl.querySelector(".count-select");
    const video = panelEl.querySelector(".video-player");
    const prevBtn = navEl.querySelector(".prev-btn");
    const nextBtn = navEl.querySelector(".next-btn");

    players.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.name;
      playerSelect.appendChild(opt);
    });

    const state = { playerId: players.length ? players[0].id : null, count: initialCount };

    function populateCountSelect(playerId) {
      countSelect.innerHTML = "";
      const list = spikesByPlayer[playerId] || [];
      list.forEach((s) => {
        const opt = document.createElement("option");
        opt.value = s.player_spike_number;
        opt.textContent = `${s.player_spike_number}回目`;
        countSelect.appendChild(opt);
      });
      countSelect.disabled = list.length === 0;
    }

    function updateNavButtons(playerId, count) {
      prevBtn.disabled = !findSpike(playerId, count - 1);
      nextBtn.disabled = !findSpike(playerId, count + 1);
    }

    function loadSpike(playerId, count) {
      const spike = findSpike(playerId, count);
      if (!spike) {
        video.removeAttribute("src");
        video.load();
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        return;
      }
      state.playerId = playerId;
      state.count = count;
      countSelect.value = String(count);
      video.src = `/media/${sessionId}/${spike.video_path}`;
      video.load();
      updateNavButtons(playerId, count);
    }

    playerSelect.addEventListener("change", () => {
      const playerId = parseInt(playerSelect.value, 10);
      populateCountSelect(playerId);
      loadSpike(playerId, 1);
    });

    countSelect.addEventListener("change", () => {
      loadSpike(state.playerId, parseInt(countSelect.value, 10));
    });

    prevBtn.addEventListener("click", () => loadSpike(state.playerId, state.count - 1));
    nextBtn.addEventListener("click", () => loadSpike(state.playerId, state.count + 1));

    if (state.playerId !== null) {
      playerSelect.value = String(state.playerId);
      populateCountSelect(state.playerId);
      const list = spikesByPlayer[state.playerId] || [];
      const startCount = list.some((s) => s.player_spike_number === initialCount) ? initialCount : 1;
      loadSpike(state.playerId, startCount);
    }

    return { video };
  }

  const panelA = createPanel("a", 1);
  const panelB = createPanel("b", 2);

  const syncBtn = document.getElementById("sync-play-btn");
  syncBtn.addEventListener("click", () => {
    [panelA.video, panelB.video].forEach((video) => {
      if (!video.src) return;
      video.currentTime = 0;
      video.play().catch(() => {});
    });
  });
});
