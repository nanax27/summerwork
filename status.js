document.addEventListener("DOMContentLoaded", () => {
  const box = document.getElementById("status-box");
  const sessionId = box.dataset.sessionId;
  const statusMessage = document.getElementById("status-message");
  const progressMessage = document.getElementById("progress-message");

  const STAGE_LABEL = { detect: "ボール検出中", extract: "動画切り抜き中" };

  async function poll() {
    let data;
    try {
      const res = await fetch(`/session/${sessionId}/status`);
      data = await res.json();
    } catch (err) {
      progressMessage.textContent = "通信エラー。再試行します...";
      setTimeout(poll, 3000);
      return;
    }

    if (data.status === "done") {
      statusMessage.textContent = `解析完了！検出スパイク数：${data.spike_count}`;
      progressMessage.innerHTML = `<a href="/session/${sessionId}/player">スパイク動画を見る</a>`;
      return;
    }
    if (data.status === "error") {
      statusMessage.textContent = "解析中にエラーが発生しました。";
      progressMessage.textContent = data.error_message || "";
      return;
    }

    statusMessage.textContent = "解析中...";
    if (data.progress) {
      const label = STAGE_LABEL[data.progress.stage] || data.progress.stage;
      const { current, total } = data.progress;
      const pct = total ? Math.min(100, Math.round((current / total) * 100)) : 0;
      progressMessage.textContent = `${label}: ${current}/${total} (${pct}%)`;
    } else {
      progressMessage.textContent = "";
    }
    setTimeout(poll, 2000);
  }

  poll();
});
