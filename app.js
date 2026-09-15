document.addEventListener("DOMContentLoaded", () => {
  const numPlayersInput = document.getElementById("num_players");
  const playerFields = document.querySelectorAll(".player-field");

  function updateVisibleFields() {
    const n = parseInt(numPlayersInput.value, 10) || 0;
    playerFields.forEach((field) => {
      const index = parseInt(field.dataset.index, 10);
      field.style.display = index <= n ? "" : "none";
    });
  }

  numPlayersInput.addEventListener("input", updateVisibleFields);
  updateVisibleFields();
});
